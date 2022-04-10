import os
import pandas as pd
import pickle

def tokenize(seq):
    token_seq = [token_dict[aa] for aa in seq]
    if len(token_seq) >= SEQ_LEN:
        return token_seq[0:SEQ_LEN]
    else:
        return token_seq + [0]*(600-len(token_seq))


SEQ_LEN = 600

aaseq_dict = pd.read_pickle(os.path.join('/home/cz332/database/resource/aaseq', 'aaseq_dict.pkl'))

token_dict = {'A': 1,
              'C': 2,
              'D': 3,
              'E': 4,
              'F': 5,
              'G': 6,
              'H': 7,
              'I': 8,
              'K': 9,
              'L': 10,
              'M': 11,
              'N': 12,
              'P': 13,
              'Q': 14,
              'R': 15,
              'S': 16,
              'T': 17,
              'V': 18,
              'W': 19,
              'Y': 20}

token_seq_dict = {}
for ec in aaseq_dict:
    token_seq_dict[ec] = {}
    genes = aaseq_dict[ec]
    for gene in genes:
        print(ec, gene)
        seq = genes[gene]
        token_seq = tokenize(seq)
        token_seq_dict[ec][gene] = token_seq

with open(os.path.join('/home/cz332/database/resource/aaseq', 'token_aaseq_dict.pkl'), 'wb') as handle:
    pickle.dump(token_seq_dict, handle)