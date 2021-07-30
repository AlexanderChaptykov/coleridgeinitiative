from typing import List
import pandas as pd
from cleantext import clean
import numpy as np

from word_river.train_data.utils import clean_text


def get_augs(training_args, data_args) -> List[str]:
    new_ds = pd.read_csv(training_args.augs_csv).title.tolist()

    df = pd.read_csv(data_args.ds_dir / 'train.csv')
    all_labels = set(df.dataset_title) | set(df.dataset_label)
    all_labels = set([clean_text(x) for x in all_labels])  # all dataset_labels and titles

    new_ds = [x for x in new_ds if not x in all_labels]

    preps = {'of', 'on', 'or', 'be', 'to', 'an', 'as', 'at', 'by', 'in', 'is', 'it', 'the', 'and', 'are', 'all', 'for',
             'from'}

    def add_capital(txt):
        return ' '.join([w.capitalize() if w not in preps else w for w in txt.strip().split()])

    return [add_capital(x) for x in new_ds]


def get_weights(train_items):
    distr = pd.Series([x.dataset_title for x in train_items]).value_counts(dropna=False)

    def filter_(v, i):
        if i is np.nan:
            return 3000
        return 300 if v > 300 else v

    probs = np.array([filter_(v, i) for v, i in zip(distr.values, distr.index)]) / distr.values
    prob_map = dict(zip(distr.index, probs))
    return [prob_map[x.dataset_title] if x.dataset_title else prob_map[np.nan] for x in train_items]


def cleaning(t):
    c_setup = {
        'no_line_breaks': True,
        'lower': False,
        'to_ascii': True,
        'fix_unicode': False,
        'lang': "en"
    }
    return clean(t, **c_setup)


