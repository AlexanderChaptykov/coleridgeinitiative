import json
from pathlib import Path

from word_river.cli_parser.dtypes import ModelArguments, DataArguments, MyTrainingArguments, WandbArguments
from transformers import HfArgumentParser

# TrainingArguments = TrainingArguments
parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments, WandbArguments))
model_args: ModelArguments
data_args: DataArguments
training_args: MyTrainingArguments
wandb_args: WandbArguments

model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()



if data_args.test_mode:
    data_args.n_data_points = 10

data_args.ds_dir = Path(data_args.ds_dir)

assert data_args.sents_win >= data_args.step_size, print(data_args.sents_win, data_args.step_size)
