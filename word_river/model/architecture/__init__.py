import logging

import torch
from torch import nn
from transformers import AutoModel, BatchEncoding, AutoConfig

from word_river.model.architecture.layers import Get_dim
from word_river.model.architecture.transformer import TransformerBlock


# noinspection PyArgumentList,PyArgumentList,PyUnusedLocal

class AbstractTransfModel(torch.nn.Module):
    """
        model = MyModel(model_args, data_args)
        t_conf = {
            'truncation': True,
            'padding': "max_length",
            'max_length': 10,
            'return_tensors': 'pt'
        }
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        texts = ['this study used data', 'from the national education longitudinal study', 'examine the effects of dual enrollment programs']
        x = tokenizer(texts, **t_conf)
        collator = Collator(data_args, model_args)
        x = collator(texts)
        out = model(x)
        # out: tuple(torch.Size([bs, token_num]), torch.Size([bs, token_num]))
    """

    def __init__(self, model_args, data_args):
        super().__init__()
        self.model_args = model_args
        self.data_args = data_args
        self.transformer = AutoModel.from_pretrained(model_args.model_name_or_path)
        conf = AutoConfig.from_pretrained(model_args.model_name_or_path)
        self.emb_size = conf.hidden_size


class MyModel(AbstractTransfModel):
    """
        model = MyModel(model_args, data_args)
        t_conf = {
            'truncation': True,
            'padding': "max_length",
            'max_length': 10,
            'return_tensors': 'pt'
        }
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        texts = ['this study used data', 'from the national education longitudinal study', 'examine the effects of dual enrollment programs']
        x = tokenizer(texts, **t_conf)
        collator = Collator(data_args, model_args)
        x = collator(texts)
        out = model(x)
        # out: tuple(torch.Size([bs, token_num]), torch.Size([bs, token_num]))
    """

    def __init__(self, model_args, data_args):
        super().__init__(model_args, data_args)

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

    def forward(self, inputs: BatchEncoding):
        """
        Usual torch forward function

        Arguments:
            tokens {torch tensor} -- Sentence tokens
            token_type_ids {torch tensor} -- Sentence tokens ids

        """

        outs = self.transformer(**inputs)
        self.batch_attns = outs.attentions

        features = outs.last_hidden_state
        features = nn.Dropout2d(p=0.2)(features)

        # print(features.shape)
        # logits = self.logits(features)
        # print(logits.shape)
        # start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
        # print(start_logits.shape)
        start_logits = self.left_way(features)
        end_logits = self.right_way(features)

        return start_logits, end_logits


class Attention(torch.nn.Module):
    def __init__(self, emb_size, mask=None,
                 w_regularizer=None, b_regularizer=None,
                 w_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        super(self.__class__, self).__init__()
        #         self.supports_masking = True
        #         self.init = initializers.get('glorot_uniform')

        #         self.W_regularizer = regularizers.get(W_regularizer)
        #         self.b_regularizer = regularizers.get(b_regularizer)

        #         self.W_constraint = constraints.get(W_constraint)
        #         self.b_constraint = constraints.get(b_constraint)
        self.linear = torch.nn.Linear(emb_size, 1)

    def forward(self, x, verbose=1, mask=None):
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)
        logging.debug(f'   x.shape {x.shape}')
        attn_weights = self.linear(x)

        if mask is not None:
            mask = torch.unsqueeze(mask, dim=-1)
            logging.debug(f'   attn_weights {attn_weights.shape}', )
            logging.debug(f'   mask {mask.shape}', )
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)

        logging.debug(f'   attn_weights {attn_weights.shape}', )
        attn_weights.transpose_(1, 2)
        logging.debug(f'   attn_weights {attn_weights.shape}')
        attn_weights = torch.softmax(attn_weights, axis=2)
        out = torch.bmm(attn_weights, x)[:, 0, :]
        return out


def load_model(device, path):
    model = MyModel()
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model
