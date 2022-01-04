from constants import *
#%%
import pandas as pd
import logging
import sys


# load pickle files
def load_kegg_mols():
    return pd.read_pickle(KEGG_MOL_PATH)


def load_reaxys_mols():
    return pd.read_pickle(REAXYS_MOL_PATH)


def load_drug_mols():
    return pd.read_pickle(DRUG_MOL_PATH).dropna(subset=['merge_id'])

def load_mol_fps():
    return pd.read_pickle(MOL_FPS_PATH)


def load_kegg_rxns():
    return pd.read_pickle(KEGG_RXN_PATH).dropna(subset=['react', 'prod'])


def load_kegg_net():
    return pd.read_pickle(KEGG_NET_PATH)


def load_reaxys_net():
    return pd.read_pickle(REAXYS_NET_PATH)


def load_reaxys_x_net():
    return pd.read_pickle(REAXYS_X_NET_PATH)


def load_hybrid_net():
    return pd.read_pickle(HYBRID_NET_PATH)


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


logging.basicConfig(
    # filename='main.log',
    # filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    # level=logging.INFO,
)

logger = logging.getLogger(__name__)
