from itertools import chain
from typing import List, Dict
import transformers

transformers.models.roberta.modeling_roberta.RobertaModel

import pandas as pd
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, BatchEncoding
import numpy as np
from word_river.model.metrics import qa_loss_fn, compute_fbeta
from word_river.train_data.utils import clean_text


class ModelPl(pl.LightningModule):
    def __init__(self, model_class, model_args, data_args, training_args, batch_aggregator_fn, lr=None):
        super().__init__()
        self.training_args = training_args
        # lr or learning_rate argument is required for auto_lr_find
        self.lr = lr
        self.data_args = data_args
        self.model_args = model_args
        self.model = model_class(model_args, data_args)
        self.criterion = qa_loss_fn
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.metric = compute_fbeta
        self.batch_aggregator_fn = batch_aggregator_fn
        self.all_doc_ids = []
        self.labels = []
        self.labels_val = []
        self.ep_num = 1
        self.preds = []
        self.preds_val = []

        self.pubs = []
        self.pubs_val = []

    def forward(self, x: BatchEncoding):
        return self.model(x['x'])

    def training_step(self, batch: Dict, batch_idx):
        return self.shared_step(batch)

    def training_epoch_end(self, outs):
        self.shared_epoch_end(outs, 'train')

    def validation_step(self, batch: Dict, batch_idx, dataset_idx=None):
        return self.shared_step(batch, dataset_idx=dataset_idx)

    def validation_epoch_end(self, outs):
        self.shared_epoch_end(outs, 'val')

    def shared_step(self, batch, loss_count=True, dataset_idx=None):
        preds = self.predict_batch(batch, loss_count)
        metrics = self.batch_metrics(preds['labels'], preds['preds'])
        if dataset_idx is not None:
            return {**preds, **metrics, **{'dataset_idx': dataset_idx}}
        return {**preds, **metrics}

    def shared_epoch_end(self, outs, epoch_type, callb=0):

        if type(outs[0]) == list:
            a_ll_m = []
            for i_, dl_out in enumerate(outs):
                outs_parsed = dict()
                for k in ['pub_title', 'preds', 'scores', 'labels']:
                    outs_parsed[k] = list(chain(*[x[k] for x in dl_out]))

                for k in ['fp', 'fn', 'tn', 'tp']:
                    outs_parsed[k] = [x[k] for x in dl_out]

                if dl_out[0]['loss']:
                    mean_loss = np.mean([x['loss'].cpu() for x in dl_out])
                else:
                    mean_loss = None

                tp, fp, fn = sum(outs_parsed['tp']), sum(outs_parsed['fp']), sum(outs_parsed['fn'])

                metrics = self.fbeta_prec_recall(tp, fp, fn)
                metrics['loss'] = mean_loss
                a_ll_m.append(metrics)
                self.log_dict({f'{k}_{epoch_type}_{i_}': metrics[k] for k in metrics}, prog_bar=True)
            self.log('fbeta_mean', np.mean([x['fbeta'] for x in a_ll_m]), prog_bar=True)
        else:
            outs_parsed = dict()
            for k in ['pub_title', 'preds', 'scores', 'labels']:
                outs_parsed[k] = list(chain(*[x[k] for x in outs]))

            for k in ['fp', 'fn', 'tn', 'tp']:
                outs_parsed[k] = [x[k] for x in outs]

            if outs[0]['loss']:
                mean_loss = np.mean([x['loss'].cpu() for x in outs])
            else:
                mean_loss = None

            tp, fp, fn = sum(outs_parsed['tp']), sum(outs_parsed['fp']), sum(outs_parsed['fn'])

            metrics = self.fbeta_prec_recall(tp, fp, fn)
            metrics['loss'] = mean_loss

            self.log_dict({f'{k}_{epoch_type}': metrics[k] for k in metrics}, prog_bar=True)

    def batch_metrics(self, y_true: List[str], preds: List[str]) -> Dict:
        # return {'tp': 3, 'fp':3}...
        all_keys = {'tp', 'fp', 'tn', 'fn'}

        def _jaccard_similarity(Y: str, Y_hat: str) -> float:
            Y, Y_hat = clean_text(Y), clean_text(Y_hat)
            if Y == None:
                if Y_hat == None:
                    return 'tn'
                else:
                    return 'fp'
            if Y_hat == None:
                return 'fn'
            a = set(Y.split())
            b = set(Y_hat.split())
            c = a.intersection(b)
            jaccard = float(len(c)) / (len(a) + len(b) - len(c))
            return 'tp' if jaccard > .5 else 'fp'

        m = [_jaccard_similarity(y, y_hat) for y, y_hat in zip(y_true, preds)]
        m = pd.Series(m).value_counts().to_dict()

        for k_ in all_keys - set(m.keys()):
            m[k_] = 0
        return m

    def predict_batch(self, batch, loss_count=True):
        start_logits, end_logits = self(batch)
        preds = self.logits2str(start_logits, end_logits, batch['x'])
        scores = list(
            (start_logits.detach().cpu().numpy().max(axis=1) + end_logits.detach().cpu().numpy().max(axis=1)) / 2)

        # example of loss - tensor(10.7982, device='cuda:0')
        if 'labels' in batch:
            if loss_count:
                loss = self.criterion(
                    start_logits,
                    end_logits,
                    batch['start_pos'],
                    batch['end_pos'],
                    batch['labels'],
                    label_w=.1,
                    non_label_w=1
                )
                return {
                    'pub_title': batch['pub_titles'],
                    'preds': preds,
                    'scores': scores,
                    'labels': batch['labels'],
                    'loss': loss
                }
            else:
                return {
                    'pub_title': batch['pub_titles'],
                    'preds': preds,
                    'scores': scores,
                    'labels': batch['labels'],
                    'loss': None
                }
        else:
            return {
                'pub_title': batch['pub_titles'],
                'preds': preds,
                'scores': scores,
            }

    # def predict_batch(self, batch):
    #     start_logits, end_logits = self(batch)
    #     preds = self.logits2str(start_logits, end_logits, batch['x'])
    #     scores = list((start_logits.cpu().numpy().max(axis=1) + end_logits.cpu().numpy().max(axis=1))/2)
    #     return {
    #         'pub_title': batch['pub_titles'],
    #         'preds': preds,
    #         'scores': scores,
    #         'labels':batch['labels']
    #     }

    def logits2str(self, start_logits, end_logits, x) -> List[str]:
        start_ids = start_logits.argmax(axis=1)
        end_ids = end_logits.argmax(axis=1)
        ranges = zip(start_ids, end_ids)
        # TODO: есть идея сделать не через декодинг а через мапинг в индекс чары
        preds = [self.tokenizer.decode(x[rng[0]:rng[1] + 1], skip_special_tokens=True) for x, rng in
                 zip(x['input_ids'], ranges)]
        preds = [clean_text(x) for x in preds]
        return [x if x else None for x in preds]

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr or self.training_args.lr_1)
        optimizer = torch.optim.Adam(
            [
                {'params': self.model.transformer.parameters(), 'lr': self.training_args.lr_1},
                {'params': self.model.left_way.parameters(), 'lr': self.training_args.lr_2},
                {'params': self.model.right_way.parameters(), 'lr': self.training_args.lr_2}
            ]
        )
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, .9, verbose=True)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": self.training_args.callbacks_monitor,
        }

    def fbeta_scor(self, tp, fp, fn, beta=.5):
        tp_beta = tp * (1 + beta ** 2)
        fn_beta = fn * (beta ** 2)
        return tp_beta / (tp_beta + fp + fn_beta)

    def fbeta_prec_recall(self, tp, fp, fn):
        d = dict()
        d['fbeta'] = round(self.fbeta_scor(tp, fp, fn), 3)
        try:
            d['precision'] = round(tp / (tp + fp), 3)
        except:
            d['precision'] = 0
        d['recall'] = round(tp / (tp + fn), 3)
        d['tp'] = tp
        d['fp'] = fp
        d['fn'] = fn
        return d
