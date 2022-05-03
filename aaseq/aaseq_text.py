import os
import pandas as pd
import logging
import pickle
from constants import *


def _convert_to_text(df):
    seq = str()
    for i, row in df.iterrows():
        aa = row[row == 1].index[0]
        seq += aa
    return seq


files = sorted(os.listdir(AASEQ_DB_PATH))

aaseq_dict = {}
for ec_pkl in files:
    ec = ec_pkl[0:-4]
    genes = pd.read_pickle(os.path.join(AASEQ_DB_PATH, '{}.pkl'.format(ec)))
    aaseq_dict[ec] = {}
    for gene in genes:
        print(ec, gene)
        aaseq_dict[ec][gene] = _convert_to_text(genes[gene])

    with open(os.path.join('/home/cz332/database/resource/aaseq', 'aaseq_dict.pkl'), 'wb') as handle:
        pickle.dump(aaseq_dict, handle)