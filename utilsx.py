from constants import *
#%%
import pandas as pd
import logging
import signal
import sys
import numpy as np

from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols

# load pickle files
def load_kegg_mols():
    return pd.read_pickle(KEGG_MOL_PATH)


def load_reaxys_mols():
    return pd.read_pickle(REAXYS_MOL_PATH)


def load_drug_mols():
    return pd.read_pickle(DRUG_MOL_PATH).dropna(subset=['merge_id'])


def load_mol_fps():
    return pd.read_pickle(MOL_FPS_PATH)


def load_enzymes():
    return pd.read_pickle(ENZYME_PATH)


def load_msc_pairs():
    return pd.read_pickle(MCS_PAIRS_PATH)


def load_kegg_rxns():
    return pd.read_pickle(KEGG_RXN_PATH).dropna(subset=['react', 'prod'])


def load_single_kegg_rxns():
    return pd.read_pickle(KEGG_SINGLE_RXN_PATH)


def load_kegg_net():
    return pd.read_pickle(KEGG_NET_PATH)


def load_reaxys_net():
    return pd.read_pickle(REAXYS_NET_PATH)


def load_reaxys_x_net():
    return pd.read_pickle(REAXYS_X_NET_PATH)


def load_hybrid_net():
    return pd.read_pickle(HYBRID_NET_PATH)


def load_mcs_pairs():
    return pd.read_pickle(MCS_PAIR_PATH)


def load_pred_mol_db():
    return pd.read_pickle(PRED_MOL_DB_PATH)


def load_pred_mol_history():
    return pd.read_pickle(PRED_MOL_HISTORY_PATH)


def load_pred_rxn_db():
    return pd.read_pickle(PRED_RXN_DB_PATH)


def load_pred_rxn_history():
    return pd.read_pickle(PRED_RXN_HISTORY_PATH)


def load_ec_gene_pair():
    return pd.read_pickle(EC_GENE_PAIR_PATH)


def load_ec_gene_token():
    return pd.read_pickle(EC_GENE_TOKEN_PATH)


def load_token_seq():
    return pd.read_pickle(TOKEN_SEQ_PATH)


def load_aaseq_text():
    return pd.read_pickle(AASEQ_TEXT_PATH)


def mol_merge_reaxys(mols, source_mols):
    reaxys_mols = []
    for mol in mols:
        if mol in source_mols.index:
            if type(source_mols.loc[mol]['merge_reaxys_id']) == list:
                reaxys_mols += source_mols.loc[mol]['merge_reaxys_id']
            else:
                reaxys_mols += [mol]
    return reaxys_mols


def load_excluded_metabolites():
    _load_metabolite_lib = lambda sheet_name: set(pd.read_excel(
        METABOLITE_LIB_PATH, sheet_name=sheet_name)['KEGG ID'].dropna())

    [free_metabolite,
     cofactor
     ] = map(_load_metabolite_lib, ['free metabolites',
                                    'cofactors'
                                    ])
    return free_metabolite.union(cofactor)


def load_simcomp():
    return pd.read_pickle(SIMCOMP_PATH)


def load_token_seq():
    return pd.read_pickle(TOKEN_SEQ_PATH)


logging.basicConfig(
    # filename='main.log',
    # filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    # level=logging.DEBUG,
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def timeout(func, input, time_out):
    class TimeoutError(Exception):
        pass
    def handler(signum, frame):
        raise TimeoutError()
    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(time_out)
    try:
        output = func(input)
    except TimeoutError:
        logger.debug(str(func), ' gets time out. The function is terminated.')
        output = False
    finally:
        signal.alarm(0)
    return output


def compute_jaccard(mol1, mol2):
    fps1 = FingerprintMols.FingerprintMol(mol1)
    fps2 = FingerprintMols.FingerprintMol(mol2)
    try:
        return DataStructs.FingerprintSimilarity(fps1,fps2)
    except Exception:
        return np.nan