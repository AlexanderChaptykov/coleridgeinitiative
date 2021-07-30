import os
from pathlib import Path
from typing import Optional

from transformers import HfArgumentParser
from dataclasses import dataclass, field

path: Path = Path(__file__).parent.parent


@dataclass
class DatasetDataArguments:
    """
    Arguments pertaining to what datasets we are going to input our model for training and eval.
    """

    out_folder: str = field(
        metadata={"help": "Folder where to save dataset. Example- datasets/18_12_20"}
    )
    model_name_or_path: str = field(
        default='roberta-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"
                          "bert-base-uncased roberta-base distilbert-base-uncased"
                  }
    )
    data_dir: lambda p: Path(p).absolute() = field(
        default=path / "artefacts",
        metadata={
            "help": "The input datasets dir. Should contain the .tsv files (or other datasets files) for the task."}
    )

    json_dir: str = field(
        default="json/18_12_2020",
        metadata={"help": "Dir with annotations and interpetations jsons."}
    )
    test_id_file: str = field(
        default="json/test_ids.csv",
        metadata={"help": "File contains test ids."}
    )
    max_source_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "ONLY IF USING MiddleTrimer"
                    "The maximum total input sequence length"
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
            )
        },

    )


parser = HfArgumentParser(DatasetDataArguments)
args = parser.parse_args()

if (
        os.path.exists(args.data_dir / args.out_folder)
        and len(os.listdir(args.data_dir / args.out_folder)) > 0
        and not args.overwrite_output_dir
):
    raise ValueError(
        f"Output directory ({args.out_folder}) already exists and "
        f"has {len(os.listdir(args.data_dir / args.out_folder))} items). "
        "Use --overwrite_output_dir to overcome."
    )
