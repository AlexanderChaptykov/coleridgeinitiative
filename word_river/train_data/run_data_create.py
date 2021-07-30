import json
import multiprocessing as mp
import time

import pandas as pd
from tqdm import tqdm

from word_river.cli_parser.train import data_args
from word_river.train_data.utils import read_json, txt_sum, Ranger, clean_text

if __name__ == '__main__':

    df = pd.read_csv(data_args.ds_dir / 'train.csv')
    all_labels = set(df.dataset_title) | set(df.dataset_label)

    with mp.Pool(mp.cpu_count()) as p:
        df['text'] = p.map(read_json, df.Id)

    df = df.groupby('text', as_index=False).agg({
        'pub_title': set,
        'Id': set,
        'text': txt_sum,
        'dataset_title': set,
        'dataset_label': set
    })

    # pub_title ?????? ?????, ????? ?????? ????????
    df['pub_title'] = [x[0] for x in df['pub_title']]

    # start_time = time.time()
    # with mp.Pool(mp.cpu_count()) as p:
    #     df['char_ranges'] = p.map(Ranger(all_labels), df.text)
    # print("--- %s Ranger seconds ---" % (time.time() - start_time))

    for i in tqdm(df.index):
        d = df.loc[i].to_dict()
        d['Id'] = list(d['Id'])
        d['dataset_title'] = list(d['dataset_title'])
        d['dataset_label'] = list(d['dataset_label'])
        labels_in_text = [d['text'][slice(*r)] for r in d['char_ranges']]
        d['bar_targets'] = '|'.join(set([clean_text(lbl) for lbl in labels_in_text]))
        with open(data_args.ds_dir / data_args.json_dir / f'{i}.json', 'w') as outfile:
            json.dump(d, outfile)
