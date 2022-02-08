from mol_pred import *

self = MoleculePrediction()

mcs_pairs = []
for m in self.simcomp_mols.index:
    for mm in self.simcomp_mols.loc[m]:
        mid = int(str(m)[1::])
        mmid = int(str(mm)[1::])
        if mid < mmid:
            mcs_pairs.append((m, mm))
        else:
            mcs_pairs.append((mm, m))

mcs_rxns = []
for r in self.single_rxns.index:
    m = self.single_rxns.loc[r]['react']
    mm = self.single_rxns.loc[r]['prod']
    mid = int(str(m)[1::])
    mmid = int(str(mm)[1::])
    if mid < mmid:
        mcs_rxns.append((m, mm))
    else:
        mcs_rxns.append((mm, m))

mcs_all = mcs_pairs+mcs_rxns
mcs_all = set(mcs_all)

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import Descriptors

valid_pairs = []
NUM = []
for (m1, m2) in mcs_all:
    try:
        mm1 = Chem.MolFromSmiles(self.mols.loc[m1].SMILES)
        mm2 = Chem.MolFromSmiles(self.mols.loc[m2].SMILES)

        mm1 = Chem.AddHs(mm1)
        mm2 = Chem.AddHs(mm2)

        numatom1 = mm1.GetNumAtoms()
        numatom2 = mm2.GetNumAtoms()

        numatom = min(numatom1, numatom2)
        NUM.append(numatom)
        valid_pairs.append((m1, m2))
    except:
        pass

print(len(valid_pairs))

import pandas as pd
data = pd.DataFrame(data={'NUM_ATOM':NUM},index=valid_pairs)
data.sort_values('NUM_ATOM',ascending=True,inplace=True)

import pickle

MCS = []
valid_pairs = []
for i,(m1,m2) in enumerate(data.index):
    try:
        logger.debug('INDEX: {}, MOL1: {}, MOL2: {}'.format(i, m1, m2))
        mm1 = Chem.MolFromSmiles(self.mols.loc[m1].SMILES)
        mm2 = Chem.MolFromSmiles(self.mols.loc[m2].SMILES)
        
        mm1 = Chem.AddHs(mm1)
        mm2 = Chem.AddHs(mm2)

        mcs = rdFMCS.FindMCS([mm1, mm2], timeout=5)
        smarts = mcs.smartsString
        MCS.append(smarts)
        valid_pairs.append((m1,m2))

        with open('mcs.pkl', 'wb') as handle:
            pickle.dump(MCS, handle)

        with open('valid_pairs.pkl', 'wb') as handle:
            pickle.dump(valid_pairs, handle)
    except:
        pass