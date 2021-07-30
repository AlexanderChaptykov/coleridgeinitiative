from typing import Tuple, Optional, List, Set

from pydantic import BaseModel, BaseConfig
from torch import Tensor
from transformers import BatchEncoding


class Publication(BaseModel):
    pub_title: str
    text: str
    dataset_titles: Optional[Set[str]]
    dataset_labels: Optional[Set[str]]

    def __repr__(self):
        text_r = f'{self.text[:60]}... text_len: {len(self.text)}'
        return f"Publication( \n" \
               f"  pub_title='{self.pub_title}', \n" \
               f"  text='{text_r}', \n" \
               f"  dataset_titles={self.dataset_titles}), \n" \
               f"  dataset_titles={self.dataset_labels})"


class Item(BaseModel):
    pub_title: str
    dataset_title: Optional[str]
    dataset_label: Optional[str]
    text: str
    char_range: Optional[Tuple[int, int]]

    def __repr__(self):
        if len(self.text) > 60:
            text_r = f'{self.text[:60]}...'
        else:
            text_r = self.text
        return f"Item( \n" \
               f"  pub_title='{self.pub_title}', \n" \
               f"  dataset_title={self.dataset_title} \n" \
               f"  dataset_label={self.dataset_label} \n" \
               f"  text='{text_r}', \n" \
               f"  char_range={self.char_range})"


class Batch(BaseModel):
    pub_titles: List[str]
    x: BatchEncoding
    start_pos: Tensor
    end_pos: Tensor
    labels: List[str]

    class Config(BaseConfig):
        arbitrary_types_allowed = True
