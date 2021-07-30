import random
from typing import Dict, Tuple, List, Union

import torch
import torch.nn.functional as F
from transformers import BartTokenizer, AutoTokenizer, BatchEncoding

from word_river.cli_parser.dtypes import DataArguments, ModelArguments
from word_river.dtypes import Item


class Collator:
    """
        collator = Collator(data_args, model_args)
    """

    def __init__(self, data_args: DataArguments, model_args: ModelArguments,
                 all_labels: Union[List[str], None] = None,
                 augmentation_list: Union[List[str], None] = None,
                 augmentation_prob=0,
                 add_false_label_prob=0,
                 tpu_num_cores=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.data_args = data_args
        self.pad_token_id = self.tokenizer.pad_token_id

        self.augmentation_list = augmentation_list
        if self.augmentation_list:
            self.augmentation_prob = augmentation_prob
        else:
            self.augmentation_prob = 0

        self.all_labels = all_labels
        if self.all_labels:
            self.add_false_label_prob = add_false_label_prob
        else:
            self.add_false_label_prob = 0

        assert (
                self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        #         self.args = args
        self.tpu_num_cores = tpu_num_cores
        add_prefix_space = {"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {}
        self.t_conf = {
            **{
                'truncation': True,
                'padding': "max_length",
                'max_length': data_args.max_source_length,
                'return_tensors': 'pt',
                'return_offsets_mapping': True,
            },
            **add_prefix_space}

        self.pred_conf = {
            **{
                'truncation': True,
                'padding': "max_length",
                'max_length': data_args.max_source_length,
                'return_tensors': 'pt',
                'return_offsets_mapping': False,
            },
            **add_prefix_space}

    def __call__(self, batch: List[Item]) -> Dict:
        if batch[0].char_range:
            return self.train_batch(batch)
        else:
            return self.pred_batch(batch)

    def train_batch(self, batch: List[Item]) -> Dict:

        if self.augmentation_prob:
            batch = [self.change_label(i) for i in batch]

        if self.add_false_label_prob:
            batch = [self.add_false_label(i) for i in batch]

        texts = [x.text for x in batch]
        x = self.tokenizer(texts, **self.t_conf)
        char_ranges = [x.char_range for x in batch]

        start_pos, end_pos = self.process_targets(x, char_ranges)

        return {
            'pub_titles': [x.pub_title for x in batch],
            'x': x,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'labels': [x.dataset_label for x in batch]
        }

    def pred_batch(self, batch: List[Item]) -> Dict:

        x = self.tokenizer([x.text for x in batch], **self.pred_conf)
        return {
            'pub_titles': [x.pub_title for x in batch],
            'x': x,
        }

    def add_false_label(self, item):
        if item.dataset_label:
            return item
        if self.add_false_label_prob > random.random():
            item.text = self.rand_insert_text(item.text, random.choice(self.all_labels))
            return item

        return item

    def rand_insert_text(self, text, ins_text):
        text = text.split()
        ins_id = random.randint(0, len(text))
        return ' '.join(text[:ins_id] + ins_text.split() + text[ins_id:])

    def change_label(self, item):
        if random.random() > self.augmentation_prob or not item.dataset_title:
            return item
        new_label = random.choice(self.augmentation_list)
        rng = item.char_range
        new_rng = (rng[0], rng[0] + len(new_label))

        new_txt = item.text[:rng[0]] + new_label + item.text[rng[1] - 1:]
        # assert new_txt[new_rng[0]:new_rng[1]] == new_label
        return Item(
            pub_title=item.pub_title,
            dataset_title=item.dataset_title,
            dataset_label=new_label,
            text=new_txt,
            char_range=new_rng
        )

    def process_targets(self, x: BatchEncoding, targets: List[Tuple]):
        offset_mapping = x.pop("offset_mapping")
        target_ids = []
        for rng, offset in zip(targets, offset_mapping):
            id_a = [i for i, (a, b) in enumerate(offset) if rng[0] >= a and rng[0] <= b]
            id_b = [i for i, (a, b) in enumerate(offset) if rng[1] - 1 in range(a, b + 1)]
            if not id_a or not id_b:
                id_a, id_b = 1, 1
            else:
                id_a = id_a[0]
                id_b = id_b[0]

            target_ids.append((id_a, id_b))

        start_pos = torch.tensor([x[0] for x in target_ids], dtype=torch.int64)
        end_pos = torch.tensor([x[1] for x in target_ids], dtype=torch.int64)
        return start_pos, end_pos

    def process_text_batch(self, batch: List[str]) -> BatchEncoding:
        batch = self.tokenizer(batch, **self.t_conf)
        # TODO: trim batch, pad batch
        return batch

    def pad_batch(self, batch):
        """
        List of tensors - > torch.Size([batch_size*sents_win, max_source_length])

        """
        batch_mask = [[i for i in range(len(x['input_ids']))] for x in batch]

        pad_tensor = lambda t, typ: F.pad(
            input=t,
            pad=(0, 0, 0, self.data_args.sents_win - len(t)),
            value=self.pad_token_id if typ == 'input_ids' else 0
        )

        input_ids = [pad_tensor(x["input_ids"], 'input_ids') for x in batch]
        attention_mask = [pad_tensor(x["attention_mask"], 'attention_mask') for x in batch]
        return torch.cat(input_ids), torch.cat(attention_mask), batch_mask

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids


def trim_batch(
        input_ids,
        pad_token_id,
        attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]
