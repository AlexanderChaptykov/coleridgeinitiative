import json
import multiprocessing as mp
from itertools import chain
from typing import List, Optional, Union

import nltk
import pandas as pd
import tqdm.notebook as tq
from tqdm import tqdm

from word_river.cli_parser.train import data_args
from word_river.dtypes import Item, Publication
from word_river.train_data.utils import Ranger, clean_text

nltk.download('punkt')

def read_train_jsons():
    res = []
    for file in tqdm((data_args.ds_dir / data_args.json_dir).glob('*.json')):
        with open(file) as json_file:
            res.append(json.load(json_file))
    return res


class Spliter:
    def __init__(self, data_args, mode: str, ranger: Optional[Ranger], get_first=True, words_per_split=300, margin=5):
        """
        mode:
            'word_split' - делим по кол-ву слов в тексте, т.е. большими кусками
            'sent_tokenize - делим с помощью nltk sent_tokenize
        ranger:
            Если он есть вытаскиваем ranges таргетов
            Если его нет, то таргеты не вытаскиваем
        get_first:
            Берем только первый попавшийся тайтл, если были тайтлы до этого то не берем
        spliter = Spliter(data_args, mode='sent_tokenize', ranger=None, get_first=True)

        """
        assert mode in {'word_split', 'sent_tokenize'}
        self.mode = mode
        self.words_per_split = words_per_split
        self.margin = margin
        self.ranger = ranger
        self.get_first = get_first
        df = pd.read_csv(data_args.ds_dir / 'train.csv')
        self.lbl_2_title = dict(zip([clean_text(x) for x in df.dataset_label], df.dataset_title))
        self.lbl_2_title = {**self.lbl_2_title,
                            **dict(zip([clean_text(x) for x in df.dataset_title], df.dataset_title))}

    def __call__(self, input: Union[List[Publication], List[str], str]) -> List[Item]:
        assert (type(input) in {str, Publication}) or all(isinstance(elem, Publication) for elem in input), \
            'correct types: List[Publication], Publication, str'

        if type(input) in {str, Publication}:
            return self.get_items(input)
        with mp.Pool(mp.cpu_count()) as p:
            res = list(tq.tqdm(p.imap(self.get_items, input), total=len(input), desc='Split'))
        return list(chain(*res))

    def get_items(self, pub: Union[Publication, str]) -> List[Item]:
        """
        2 сценария работы:
            1) Если нет Ranger это инференс
            2) если есть то это predict
        """

        if type(pub) == str:
            pub_title = 'str'
            text = pub
        elif type(pub) == Publication:
            pub_title = pub.pub_title
            text = pub.text

        # just predict
        if not self.ranger:
            return [Item(pub_title=pub_title, text=txt) for txt in self.split_text(text)]

        res = []
        splited_texts = self.split_text(text)
        ranges = [self.ranger(txt) for txt in splited_texts]

        if self.get_first:
            pub_titles = set()
            for txt, rng in zip(splited_texts, ranges):
                dataset_label, dataset_title = self.get_ttl_lbl(txt, rng)

                if dataset_title in pub_titles and rng != (1, 1):
                    continue
                else:
                    item = Item(
                        pub_title=pub_title,
                        dataset_title=dataset_title,
                        dataset_label=dataset_label,
                        text=txt,
                        char_range=rng
                    )
                    pub_titles.add(dataset_title)
                    res.append(item)
        else:
            for txt, rng in zip(splited_texts, ranges):
                dataset_label, dataset_title = self.get_ttl_lbl(txt, rng)

                item = Item(
                    pub_title=pub_title,
                    dataset_title=dataset_title,
                    dataset_label=dataset_label,
                    text=txt,
                    char_range=rng
                )
                res.append(item)
        return res

    def get_ttl_lbl(self, text, rng):
        if rng == (1, 1):
            return None, None
        else:
            dataset_label = clean_text(text[slice(*rng)])

            dataset_title = self.lbl_2_title[dataset_label]
        return dataset_label, dataset_title

    def split_text(self, txt: str) -> List[str]:
        if self.mode == 'word_split':
            texts = []
            txt_split = txt.split(' ')
            for i in range(len(txt_split) // self.words_per_split + 1):
                start = max(0, self.words_per_split * i - self.margin)
                end = self.words_per_split * (i + 1) + self.margin
                texts.append(" ".join(txt_split[start: end]))
            return texts
        elif self.mode == 'sent_tokenize':
            texts = nltk.sent_tokenize(txt)
            return texts


cv_sets = [{'Advanced National Seismic System (ANSS) Comprehensive Catalog (ComCat)',
            'CAS COVID-19 antiviral candidate compounds dataset',
            'COVID-19 Deaths data',
            'COVID-19 Image Data Collection',
            'COVID-19 Precision Medicine Analytics Platform Registry (JH-CROWN)',
            'Census of Agriculture',
            'Characterizing Health Associated Risks, and Your Baseline Disease In SARS-COV-2 (CHARYBDIS)',
            'High School Longitudinal Study',
            'Program for the International Assessment of Adult Competencies',
            'SARS-CoV-2 genome sequence',
            'Sea, Lake, and Overland Surges from Hurricanes',
            'World Ocean Database'},
           {'Agricultural Resource Management Survey',
            'CAS COVID-19 antiviral candidate compounds dataset',
            'COVID-19 Deaths data',
            'Common Core of Data',
            'Early Childhood Longitudinal Study',
            'FFRDC Research and Development Survey',
            'Higher Education Research and Development Survey',
            'National Teacher and Principal Survey',
            'Survey of Earned Doctorates',
            'Survey of Graduate Students and Postdoctorates in Science and Engineering',
            'Survey of State Government Research and Development'},
           {'Aging Integrated Database (AGID)',
            'Agricultural Resource Management Survey',
            'COVID-19 Open Research Dataset (CORD-19)',
            'Education Longitudinal Study',
            'High School Longitudinal Study',
            'National Education Longitudinal Study',
            'Rural-Urban Continuum Codes',
            'School Survey on Crime and Safety',
            "The National Institute on Aging Genetics of Alzheimer's Disease Data Storage Site (NIAGADS)",
            'World Ocean Database'},
           {'Beginning Postsecondary Student',
            'COVID-19 Open Research Dataset (CORD-19)',
            'COVID-19 Precision Medicine Analytics Platform Registry (JH-CROWN)',
            'Census of Agriculture',
            'Complexity Science Hub COVID-19 Control Strategies List (CCCSL)',
            'International Best Track Archive for Climate Stewardship',
            'National Teacher and Principal Survey',
            'North American Breeding Bird Survey (BBS)',
            'Our World in Data COVID-19 dataset',
            'Survey of Industrial Research and Development',
            'Survey of Science and Engineering Research Facilities',
            "The National Institute on Aging Genetics of Alzheimer's Disease Data Storage Site (NIAGADS)"},
           {'Advanced National Seismic System (ANSS) Comprehensive Catalog (ComCat)',
            'COVID-19 Open Research Dataset (CORD-19)',
            'Characterizing Health Associated Risks, and Your Baseline Disease In SARS-COV-2 (CHARYBDIS)',
            'Higher Education Research and Development Survey',
            'NOAA Tide Gauge',
            'National Assessment of Education Progress',
            'North American Breeding Bird Survey (BBS)',
            'Optimum Interpolation Sea Surface Temperature',
            'RSNA International COVID-19 Open Radiology Database (RICORD)',
            'Trends in International Mathematics and Science Study'}]

cv_set = cv_sets[0]
