from typing import List

import numpy as np
import pandas as pd

from word_river.cli_parser.train import data_args


# nltk.download('punkt')


def cv_test_sets(
        test_sets_num=14,
        ok_range=range(2000, 2500),
        max_common_pubs_ok=600,
        max_dataset_names=2,
) -> List[set]:
    """
    получить такие тестовые датасеты которые
    1. Имеют не больше 2 общих между собой
    2. Кол-во публикаций 2000-2500
    3. Не более 600 общих публикаций с трейном
    """

    # test_sets = np.random.choice(df.dataset_title.unique(), test_sets_num)
    #
    # test_pubs = set(df[df.dataset_title.isin(test_sets)].pub_title)
    #
    # train_pubs = set(df[~df.dataset_title.isin(test_sets)].pub_title)
    df = pd.read_csv(data_args.ds_dir / 'train.csv')

    cv_test_sets = []
    for a in range(500):

        test_pubs = {}
        common_pubs_ok = False

        while not (len(test_pubs) in ok_range and common_pubs_ok):
            test_sets = np.random.choice(df.dataset_title.unique(), test_sets_num)

            test_pubs = set(df[df.dataset_title.isin(test_sets)].pub_title)

            train_pubs = set(df[~df.dataset_title.isin(test_sets)].pub_title)

            common_pubs_ok = len(test_pubs & train_pubs) < max_common_pubs_ok

        cv_test_sets.append(set(test_sets))

    best_sets = [set()]
    for _ in range(len(cv_test_sets)):
        for s in cv_test_sets:
            if all([len(b_s & s) <= max_dataset_names for b_s in best_sets]):
                best_sets.append(s)
    best_sets.remove(set())
    return best_sets


cv_sets = cv_test_sets()

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



