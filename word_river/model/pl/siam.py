from itertools import chain
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import MetricCollection, F1, Precision, Recall
from transformers import AutoTokenizer, BatchEncoding, AutoConfig, AutoModel

from word_river.model.architecture.layers import AttentionMy, Get_dim
from word_river.model.dataset import trim_batch
from word_river.train_data.utils import clean_text


# from word_river.model.pl.utils import trim_batch


class ABC_PL(pl.LightningModule):
    def __init__(self, model_class, model_args, data_args, training_args, criterion, metric):
        super().__init__()
        self.training_args = training_args
        self.model = model_class(model_args, data_args)

        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.pad_token_id = AutoConfig.from_pretrained(model_args.model_name_or_path).pad_token_id

        self.criterion = criterion
        self.metric = metric

        self.model_args = model_args
        self.data_args = data_args
        self.transformer = AutoModel.from_pretrained(model_args.model_name_or_path)
        conf = AutoConfig.from_pretrained(model_args.model_name_or_path)
        self.emb_size = conf.hidden_size

    def training_step(self, batch: Dict, batch_idx):
        return self.shared_step(batch)

    def validation_step(self, batch: Dict, batch_idx, dataset_idx=None):
        return self.shared_step(batch, dataset_idx=dataset_idx)

    def training_epoch_end(self, outs):
        self.shared_epoch_end(outs, 'train')

    def validation_epoch_end(self, outs):
        self.shared_epoch_end(outs, 'val')


class ModelPl(ABC_PL):
    def __init__(self, model_class, model_args, data_args, training_args, criterion, metric):
        super().__init__(model_class, model_args, data_args, training_args, criterion, metric)

        self.left_way = nn.Sequential(
            nn.LSTM(self.emb_size, 10, batch_first=True, bidirectional=True),
            Get_dim(0),
            nn.Tanh(),
            nn.Dropout2d(p=0.2),
            nn.Linear(20, 1),
            nn.Flatten()
        )
        self.right_way = nn.Sequential(
            nn.LSTM(self.emb_size, 10, batch_first=True, bidirectional=True),
            Get_dim(0),
            nn.Tanh(),
            nn.Dropout2d(p=0.2),
            nn.Linear(20, 1),
            nn.Flatten()
        )

        self.logits = torch.nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.Dropout(.2),
            nn.Tanh(),
            nn.Linear(self.emb_size, 2),
        )

    def forward(self, x: BatchEncoding):
        outs = self.transformer(**x)[0]
        # self.batch_attns = outs.attentions

        # features = outs.last_hidden_state
        outs = nn.Dropout2d(p=0.2)(outs)

        # print(features.shape)
        # logits = self.logits(features)
        # print(logits.shape)
        # start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
        # print(start_logits.shape)
        start_logits = self.left_way(outs)
        end_logits = self.right_way(outs)
        return start_logits, end_logits

    def shared_step(self, batch, dataset_idx=None):
        preds = self.predict_batch(batch)
        # TODO: ????? ???????? ??????? ?? ??????, ?? ????????? ????...
        metrics = self.batch_metrics(batch['labels'], preds['preds'])
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
        res = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

        def _jaccard_similarity(Y: str, Y_hat: str) -> str:
            Y, Y_hat = clean_text(Y), clean_text(Y_hat)
            a = set(Y.split())
            b = set(Y_hat.split())
            c = a.intersection(b)
            jaccard = float(len(c)) / (len(a) + len(b) - len(c))
            return 'tp' if jaccard > .5 else 'fp'

        res = {
            (None, None): 'tn',
            (None, str): 'fp',
            (str, None): 'fn',
            (str, str): _jaccard_similarity()
        }

        m = [_jaccard_similarity(y, y_hat) for y, y_hat in zip(y_true, preds)]
        m = pd.Series(m).value_counts().to_dict()

        return {**res, **m}

    def predict_batch(self, batch) -> Dict:
        start_logits, end_logits = self(batch)
        preds = self.logits2str(start_logits, end_logits, batch)
        scores = list(
            (start_logits.detach().cpu().numpy().max(axis=1) + end_logits.detach().cpu().numpy().max(axis=1)) / 2)

        # example of loss - tensor(10.7982, device='cuda:0')
        if 'labels' in batch:
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
            }

    def logits2str(self, start_logits, end_logits, batch) -> List[Union[str, None]]:
        start_ids = start_logits.argmax(axis=1)
        end_ids = end_logits.argmax(axis=1)

        start_chars = [m[ids][0].item() for ids, m in zip(start_ids, batch['offset_mapping'])]
        end_chars = [m[ids][1].item() for ids, m in zip(end_ids, batch['offset_mapping'])]

        preds = [text[a:z] for a, z, text in zip(start_chars, end_chars, batch['text'])]

        preds = [clean_text(x) for x in preds]
        return [x if x else None for x in preds]

    def configure_optimizers(self):
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


class SiamesePl(ABC_PL):

    def shared_step(self, batch, dataset_idx=None):
        batch = {k: trim_batch(batch[k], self.pad_token_id) for k in batch}
        a_out, p_out, n_out = self.model(batch)
        loss = self.criterion(a_out, p_out, n_out)
        self.log('loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {'params': self.model.parameters(), 'lr': self.training_args.lr_1},
            ]
        )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": self.training_args.callbacks_monitor,
        }


class AbstractLightningModule(pl.LightningModule):
    # model_class, model_args, data_args, training_args, criterion, metric
    def __init__(self, model_args, data_args, training_args):
        super().__init__()
        self.training_args = training_args
        self.data_args = data_args

        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.conf = AutoConfig.from_pretrained(model_args.model_name_or_path)
        self.pad_token_id = self.conf.pad_token_id
        self.emb_size = self.conf.hidden_size
        self.transformer = AutoModel.from_pretrained(model_args.model_name_or_path)

    def training_step(self, batch: Dict, batch_idx):
        return self.shared_step(batch, typ_='train')

    def validation_step(self, batch: Dict, batch_idx, dataset_idx=None):
        return self.shared_step(batch, typ_='val')


class BinClassiffierPl(AbstractLightningModule):
    # model_class, model_args, data_args, training_args, criterion, metric
    def __init__(self, model_args, data_args, training_args):
        super().__init__(model_args, data_args, training_args)
        self.metrics = MetricCollection([F1(), Precision(), Recall()])

        self.attn = AttentionMy(self.emb_size)
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.emb_size, 1),
            nn.Sigmoid(),
            nn.Flatten(start_dim=0)
        )
        self.criterion = torch.nn.BCELoss()

    def forward(self, inputs):
        outs = self.transformer(**inputs)[0]

        features = nn.Dropout2d(p=0.2)(outs)
        features = self.attn(features, verbose=False)
        y_hat = self.net(features)
        y_hat = y_hat.type(torch.float)
        return y_hat

    def shared_step(self, batch, typ_):
        x, y = batch
        x = trim_batch(x, self.pad_token_id)
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        m = self.metrics(y_hat, y.type(torch.int32))

        self.log_dict({f'{k}_{typ_}': m[k] for k in m}, prog_bar=True)
        self.log(f'loss_{typ_}', loss)

    def shared_epoch_end(self, outs, epoch_type):
        m = self.metrics.compute()
        self.log_dict({f'{k}_{epoch_type}': m[k] for k in m})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.training_args.lr_1)
        # optimizer = torch.optim.Adam(
        #     [
        #         # {'params': self.transformer.parameters(), 'lr': self.training_args.lr_1},
        #         {'params': self.attn.parameters(), 'lr': self.training_args.lr_1},
        #         # {'params': self.net.parameters(), 'lr': self.training_args.lr_1},
        #     ]
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": self.training_args.callbacks_monitor,
        }

    def training_epoch_end(self, outs):
        self.shared_epoch_end(outs, 'train')

    def validation_epoch_end(self, outs):
        self.shared_epoch_end(outs, 'val')


class TripletSiamPl(AbstractLightningModule):
    """
    TripletDs()
    """

    # model_class, model_args, data_args, training_args, criterion, metric
    def __init__(self, model_args, data_args, training_args):
        super().__init__(model_args, data_args, training_args)
        # self.metrics = MetricCollection([F1(), Precision(), Recall()])

        # TODO: check args of triplet
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)

        self.anchor_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.SELU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Softmax()
        )
        self.positive_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.SELU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Softmax()
        )
        self.negative_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.SELU(),
            nn.Linear(self.emb_size, self.emb_size),
            nn.Softmax()
        )
        self.attn_a = AttentionMy(self.emb_size)
        self.attn_p = AttentionMy(self.emb_size)
        self.attn_n = AttentionMy(self.emb_size)

    def forward(self, input):
        # {'a_toks': a_toks, 'p_toks': p_toks, 'n_toks': n_toks}
        a_embeds = self.transformer(**input['a_toks'])[0]
        p_embeds = self.transformer(**input['p_toks'])[0]
        n_embeds = self.transformer(**input['n_toks'])[0]

        a_embeds = self.attn_a(a_embeds, mask=input['a_toks']['attention_mask'], verbose=0)
        p_embeds = self.attn_p(p_embeds, mask=input['p_toks']['attention_mask'], verbose=0)
        n_embeds = self.attn_n(n_embeds, mask=input['n_toks']['attention_mask'], verbose=0)

        a_out = self.anchor_net(a_embeds)
        p_out = self.positive_net(p_embeds)
        n_out = self.negative_net(n_embeds)
        return a_out, p_out, n_out

    def shared_step(self, batch, typ_):
        a_out, p_out, n_out = self(batch)
        loss = self.criterion(a_out, p_out, n_out)
        self.log(f'loss_{typ_}', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    #
    # def training_epoch_end(self, outs):
    #     self.log('train_e_loss', torch.stack([x['loss'] for x in outs]).mean())
    #
    # def validation_epoch_end(self, outs):
    #     self.log('val_e_loss', torch.stack(outs).mean())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.training_args.lr_1)
        # optimizer = torch.optim.Adam(
        #     [
        #         # {'params': self.transformer.parameters(), 'lr': self.training_args.lr_1},
        #         {'params': self.attn.parameters(), 'lr': self.training_args.lr_1},
        #         # {'params': self.net.parameters(), 'lr': self.training_args.lr_1},
        #     ]
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": self.training_args.callbacks_monitor,
        }
