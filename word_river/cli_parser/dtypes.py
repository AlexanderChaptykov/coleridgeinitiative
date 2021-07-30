from pathlib import Path, PosixPath
from typing import Optional, List, Tuple, Union

from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which modelport/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='roberta-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"
                          "bert-base-uncased roberta-base distilbert-base-uncased"
                  }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    num_cls: int = field(
        default=2,
        metadata={"help": "How much classes"},
    )
    lstm_out_sent: int = field(
        default=100,
        metadata={"help": "Number of LSTM cells"}
    )
    kernel_size_1: int = field(
        default=1,
        metadata={"help": "1 CNN layer kernel size"}
    )
    kernel_size_2: int = field(
        default=3,
        metadata={"help": "2 CNN layer kernel size"}
    )
    kernel_size_3: int = field(
        default=5,
        metadata={"help": "3 CNN layer kernel size"}
    )
    kernel_size_4: int = field(
        default=7,
        metadata={"help": "4 CNN layer kernel size"}
    )
    filters_1: int = field(
        default=100,
        metadata={"help": "1 CNN layer neurons quantity"}
    )
    filters_2: int = field(
        default=100,
        metadata={"help": "2 CNN layer neurons quantity"}
    )
    filters_3: int = field(
        default=100,
        metadata={"help": "3 CNN layer neurons quantity"}
    )
    filters_4: int = field(
        default=100,
        metadata={"help": "4 CNN layer neurons quantity"}
    )
    activation: str = field(
        default='ELU',
        metadata={"help": "ReLU/SELU/ELU/CELU/GELU/LeakyReLU/RReLU/PReLU activation for CNN"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Some models support gradient checkpointing and require less memory"}
    )
    port: int = field(
        default=61643,
        metadata={"help": "because dont worked without ))"}
    )

@dataclass
class DataArguments:
    """
    Arguments pertaining to what datasets we are going to input our model for training and eval.
    """
    ds_dir: PosixPath = field(
        default='/Users/alexch/PycharmProjects/tmp/coleridgeinitiative-show-us-the-data_ds',
        metadata={"help": "Folder with train.csv, test.csv class_weights"}
    )
    json_dir: str = field(
        default='jsons',
        metadata={"help": "where to save jsons"}
    )
    test_mode: bool = field(
        default=False,
        metadata={
            "help": "Fast mode for debug"
        },
    )
    win: int = field(
        default=20,
        metadata={
            "help": "Size of sentence window"
        },
    )
    num_samples: int = field(
        default=None,
        metadata={
            "help": "Size of ds"
        },
    )
    data_dir: lambda p: Path(p).absolute() = field(
        default=Path(__file__).parent.parent / "artefacts/datasets",
        metadata={
            "help": "The input datasets dir. Should contain the .tsv files (or other datasets files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    sents_win: Optional[int] = field(
        default=50,
        metadata={
            "help": "Len of sentences per datapoint"
        },
    )
    step_size: Optional[int] = field(
        default=50,
        metadata={
            "help": "Step size between datapoints. Only for prediction. More s"
        },
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_workers: int = field(
        default=1,
        metadata={'help': "Number of workers for dataloaders"}
    )
    positive_sample: float = field(
        default=0.95,
        metadata={
            "help": "Positive sample percentage"
        }
    )
    n_data_points: Optional[int] = field(default=50,
                                         metadata={"help": "# training datapoints for one epoch. must be > 0"})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    class_weights = None
    class2num = None
    test_rids = None


@dataclass
class MyTrainingArguments(TrainingArguments):
    use_tpu: bool = field(
        default=False,
        metadata={"help": "Use or not TPU"}
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "If debug no train just parsing args"}
    )
    output_dir: str = field(
        default=Path(__file__).parent.parent / "artefacts/models",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    callbacks_monitor: str = field(
        default='f1_score',
        metadata={"help": "The metric watched by callbacks"}
    )
    callbacks_mode: str = field(
        default='max',
        metadata={"help": "The goal is to minimize the callbacks_monitor metric - min, or to maximize - max"}
    )

    early_stopping_patience: int = field(
        default=9,
        metadata={"help": "Early stopping patience - maximum number of epochs to train without improvement"}
    )
    scheduler_patience: int = field(
        default=7,
        metadata={
            "help": "Scheduler patience - maximum number of epochs to wait without improvement before reducing LR"}
    )
    save_top_k: int = field(
        default=1,
        metadata={"help": "How many best models to save"}
    )
    gradient_clip_val: float = field(
        default=0.0,
        metadata={"help": "Maximum gradient value. 0 - no clipping."}
    )
    log_every_n_steps: int = field(
        default=1,
        metadata={"help": "Logging frequency"}
    )
    auto_lr_find: int = field(
        default=0,
        metadata={
            "help": "Apply learning rate finder (1) or not (0). When applied all optimizers have the same learning rate."}
    )
    mixed_precision: bool = field(
        default=False,
        metadata={"help": "Use mixed precision or not."}
    )
    limit_train_batches: float = field(
        default=1.0,
        metadata={"help": "Fraction of training batches used in training"}
    )
    limit_val_batches: float = field(
        default=1.0,
        metadata={"help": "Fraction of validation batches used in validation"}
    )
    max_epochs: int = field(
        default=int(1e10),
        metadata={"help": "Maximum number of training epochs"}
    )
    max_steps: int = field(
        default=None,
        metadata={"help": "Maximum number of training steps"}
    )
    gpus: str = field(
        default=None,
        metadata={"help": "Which gpu to use (index), comma separated if multiple (1,2)"}
    )
    lr_1: float = field(
        default=3e-5,
        metadata={"help": "Words model learning rate"}
    )
    lr_2: float = field(
        default=1e-4,
        metadata={"help": "Sentence model learning rate"}
    )
    lr_3: float = field(
        default=0.2,
        metadata={"help": "Sentence predictor model learning rate"}
    )
    loss: str = field(
        default='f1',
        metadata={'help': "Loss, cross_entropy or f1"}
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={"help": "Accumulates grads every k batches."}
    )
    mean_average_pooling: int = field(
        default=0,
        metadata={"help": "Averages"}
    )
    finetune: int = field(
        default=0,
        metadata={"help": "Use finetuning - 1. Train networks 2 and 3; 2. Train network 1; 3. Train whole network."}
    )
    finetune_patience: int = field(
        default=5,
        metadata={"help": "Finetune patience - number of epochs to train without improvement before switching nets."}
    )
    save_checkpoints: int = field(
        default=1,
        metadata={"help": "0 - do not save checkpoints, 1 - save checkpoints"}
    )


@dataclass
class WandbArguments:
    project: str = field(
        default='test',
        metadata={"help": "Wandb project."}
    )
    entity: str = field(
        default='test',
        metadata={"help": "Wandb entity."}
    )

def ad() -> List[Union[str, Tuple[int, int]]]:
    return ['text of candidate', (0, 3)]
