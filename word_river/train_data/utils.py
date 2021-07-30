import json
import re
import string
from functools import reduce
from itertools import combinations, chain
from typing import List, Tuple, Union

import pandas as pd
from tqdm import tqdm

from word_river.cli_parser.train import data_args
from word_river.dtypes import Publication


def clean_text(txt):
    if type(txt) == str:
        return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower()).strip()
    elif txt is None:
        return
    else:
        assert False, f'bad type {type(txt)}'


class Ranger:
    def __init__(self, data_args, mode='get_all'):

        self.no_range = 1, 1
        df = pd.read_csv(data_args.ds_dir / 'train.csv')
        all_labels = set(df.dataset_title) | set(df.dataset_label)
        self.all_cln_labels = [clean_text(x) for x in all_labels]  # all dataset_labels and titles
        self.pattern = set(string.ascii_uppercase + string.ascii_lowercase + ''.join([str(x) for x in range(10)]))
        self.mode = mode

    def get_maper(self, text):
        norm_inds = range(len(text))
        clean_inds = []
        i = 0
        prev = False
        letter_add = False
        for l in text:
            if l in self.pattern:
                clean_inds.append(i)
                # clean_inds_.append([i, l, l])
                i += 1
                prev = False
                letter_add = True
            else:
                if not prev and letter_add:
                    clean_inds.append(i)
                    # clean_inds_.append([i, ' ', l])
                    i += 1
                    prev = True
                else:
                    clean_inds.append(None)
                    # clean_inds_.append([None, '', l])
        return dict([x for x in zip(clean_inds, norm_inds) if not (x[0] is None)])

    def suppres_kids(self, ranges: List[Tuple[int, int]]):
        """Убираем те диапазоны у которых есть "родители" """

        remove_it = []
        for rng_1, rng_2 in combinations(ranges, 2):
            set_1, set_2 = set(range(*rng_1)), set(range(*rng_2))

            if set_1 <= set_2:
                remove_it.append(rng_1)
            elif set_1 >= set_2:
                remove_it.append(rng_2)

            elif set_1 & set_2:
                print('Warning common chars', rng_1, rng_2)

        return [r for r in ranges if r not in set(remove_it)]

    def __call__(self, text) -> Union[List[Tuple[int, int]], Tuple[int, int]]:
        clean2norm = self.get_maper(text + ' ')
        cln_text = clean_text(text)
        all_ranges = []
        for cln_lbl in self.all_cln_labels:
            clean_ranges = [m.span() for m in re.finditer(cln_lbl, cln_text)]
            for cln_a, cln_z in clean_ranges:
                a, z = clean2norm[cln_a], clean2norm[cln_z]
                assert clean_text(text[a: z]) == cln_lbl
                all_ranges.append((a, z))
        all_ranges = self.suppres_kids(all_ranges)

        if not all_ranges:
            return self.no_range
        if self.mode == 'get_all':
            return all_ranges
        elif self.mode == 'get_longest':
            return max(all_ranges, key=lambda x: x[1] - x[0])


def preproces_text(text):
    return ' '.join(text.split()).replace(chr(304), 'i')


def txt_sum(series):
    return reduce(lambda x, y: x.strip() + ' ' + y.strip(), series)


def read_json(f):
    with open(data_args.ds_dir / 'train' / (f + ".json")) as f:
        json_sections = json.load(f)
    json_text = ' '.join([x['section_title'] + ' ' + x['text'] for x in json_sections])
    return preproces_text(json_text)

def check_distr(dl):
    print('Проверка распределения')
    lab_ = [x['labels'] for x in dl]
    lab_ = list(chain(*lab_))
    print('zeros/all', sum([1 for x in lab_ if x])/ len(lab_))
    print(pd.Series(lab_).value_counts(dropna=0))

def get_publications(df) -> List[Publication]:
    print('Get publications...')
    # with mp.Pool(mp.cpu_count()) as p:
    #     df['text'] = p.map(read_json, df.Id)

    df['text'] = list(map(read_json, df.Id))
    df = df[['text', 'pub_title', 'dataset_title', 'dataset_label']]
    df = df.groupby('text', as_index=False).agg({
        'pub_title': set,
        'dataset_title': set,
        'dataset_label': set
    })

    # pub_title ?????? ?????, ????? ?????? ????????
    df['pub_title'] = [list(x)[0] for x in df['pub_title']]

    publications = []

    for i in tqdm(df.index):
        pub = Publication(
            pub_title=df.loc[i]['pub_title'],
            text=df.loc[i]['text'],
            dataset_titles=df.loc[i]['dataset_title'],
            dataset_labels=df.loc[i]['dataset_label'],
        )
        publications.append(pub)

    return publications


pub_dubles = {
                 ' Graduate Enrollment in Science and Engineering Grew Substantially in the Past Decade but Slowed in 2010',
                 'Graduate Enrollment in Science and Engineering Grew Substantially in the Past Decade but Slowed in 2010. InfoBrief. NSF 12-317'},
{
    'Reading Ability Development from Kindergarten to Junior Secondary: Latent Transition Analyses with Growth Mixture Modeling',
    'Reading ability development from kindergarten to junior secondary: Latent transition analyses with growth mixture modeling'},
{
    'Mass coral bleaching due to unprecedented marine heatwave in Papahanaumokuakea Marine National Monument (Northwestern Hawaiian Islands)',
    'Mass coral bleaching due to unprecedented marine heatwave in Papah?\x81naumoku?\x81kea Marine National Monument (Northwestern Hawaiian Islands)'},
{' Reducing Wave-Induced Microwave Water-Level Measurement Error with a Least Squares',
 'Reducing Wave-Induced Microwave Water-Level Measurement Error with a Least Squares?Designed Digital Filter*'},
{'Historical Context and Ongoing Contributions',
 'National Longitudinal Studies of Kindergarten Children'},
{' Higher Education in Science and Engineering',
 ' Higher Education in Science and Engineering Science & Engineering Indicators 2020 NSB-2019-7'},
{'The Academic Success of East Asian American Youth',
 'The academic success of East Asian American youth: The role of shadow education'},
{'Facilitating the Transition to Kindergarten',
 'Facilitating the Transition to Kindergarten: What ECLS-K Data Tell Us about School Practices Then and Now'},
{'Examining the Intersectionality among Teacher Race/Ethnicity, School Context, and Risk for Occupational Stress',
 'Examining the intersectionality among teacher race/ethnicity, school context, and risk for occupational stress'},
{'Who Enrolls in High School Music? A National Profile of U.S. Students, 2009?2013',
 'Who Enrolls in High School Music? A National Profile of U.S. Students, 2009?2013:'}
