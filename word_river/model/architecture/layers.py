import logging

import torch


class Get_dim(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return inputs[self.dim]


class Apply2DMask(torch.nn.Module):
    """
    inp = torch.arange(40).view(2, 5, 4).type(torch.float)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float)

    Apply2DMask()(inp, mask)
    ... tensor([[ 4.,  5.,  6.,  7.],
                [22., 23., 24., 25.]])

    """

    def forward(self, inputs, mask):
        assert len(inputs.shape) == 3
        assert len(mask.shape) == 2

        inputs = inputs.transpose(2, 1).type(torch.float)
        mask = mask.unsqueeze(-1).type(torch.float)
        # sum
        res = torch.bmm(inputs, mask)
        # average
        res = (res * (1 / mask.sum(axis=1).unsqueeze(-1))).squeeze(2)
        return res


class AttentionMy(torch.nn.Module):
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
