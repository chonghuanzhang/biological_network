#%%
from utilsx import *
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem

import warnings
warnings.filterwarnings("ignore")

class MoleculePrediction:

    def __init__(self):

        self.mols = load_kegg_mols()

        # data from SIMCOMP api
        self.simcomp_mols = load_simcomp()

        self.rxns = load_kegg_rxns()
        # self.one2one_wire()
        self.single_rxns = load_single_kegg_rxns()

        self.rxn_pred = ReactionPrediction()

    def iter_target_mols(self):
        for mol in self.mols.index[9:100]:
            try:
                self.load_mol(mol)

            except Exception:
                pass

    def one2one_wire(self):
        self.rxns['pred_origin'] = np.nan
        self.rxns['direct'] = np.nan
        self.single_rxns = self.rxns.loc[(self.rxns['react'].map(len) == 1) & (self.rxns['prod'].map(len) == 1)]
        self.single_rxns['react'] = [mols[0] for mols in self.single_rxns['react']]
        self.single_rxns['prod'] = [mols[0] for mols in self.single_rxns['prod']]

        self.single_rxns['react_smiles'] = list(self.mols.loc[self.single_rxns['react']].SMILES)
        self.single_rxns['prod_smiles'] = list(self.mols.loc[self.single_rxns['prod']].SMILES)

    def load_mol(self, mol):
        self.mol = mol
        self.similar_mols = self.simcomp_mols[self.mol]

        for self.similar_mol in self.similar_mols:
            self.find_ref_rxn()
            if not self.ref_rxn.empty:
                for i,rxn in self.ref_rxn.iterrows():
                    logger.debug('TARGET MOL: {}, SIMILAR MOL: {}, REFERENCE RXN: {}'.format(self.mol, self.similar_mol, i))
                    self.rxn_pred.load_rxn(rxn,
                                           self.mols.loc[self.mol]['SMILES'],
                                           self.mol
                                           )

    def find_ref_rxn(self):
        reverse = self.single_rxns.loc[(self.single_rxns['react'] == self.similar_mol) & (self.single_rxns['prod'] != self.mol)]
        reverse.direct = -1

        forward = self.single_rxns.loc[(self.single_rxns['react'] != self.mol) & (self.single_rxns['prod'] == self.similar_mol)]
        forward.direct = 1

        self.ref_rxn = pd.concat([reverse, forward])
        self.ref_rxn.pred_origin = self.ref_rxn.index


class ReactionPrediction:
    def __init__(self, timeout=5, minNumAtoms=2):
        self.timeout = timeout
        self.minNumAtoms = minNumAtoms

        self.database = PredictedDatabaseRegistration()

    def load_rxn(self, rxn, mol, mol_id):
        self.mol = Chem.AddHs(Chem.MolFromSmiles(mol))
        self.mol_smiles = mol
        self.mol_id = mol_id

        self.rxn = rxn
        react = self.rxn['react_smiles']
        prod = self.rxn['prod_smiles']
        self.react = Chem.AddHs(Chem.MolFromSmiles(react))
        self.prod = Chem.AddHs(Chem.MolFromSmiles(prod))

        if self.rxn.direct == -1:
            self.reagent = self.prod
            self.similar_mol = self.react
        else:
            self.reagent = self.react
            self.similar_mol = self.prod

        self.subgraph_check()

    def compute_mcs(self, mol1, mol2):
        mcs = rdFMCS.FindMCS([mol1, mol2], timeout=self.timeout)
        mol = Chem.MolFromSmarts(mcs.smartsString)
        return mol

    def subgraph_check(self):
        try:
            self.mcs_rxn = self.compute_mcs(self.react, self.prod)
            self.mcs_mol = self.compute_mcs(self.mol, self.similar_mol)

            self.similar_mol_res = AllChem.DeleteSubstructs(self.similar_mol, self.mcs_rxn)

            if self.mcs_mol.HasSubstructMatch(self.similar_mol_res):
                self.rxn_pred()

        except Exception:
            pass

    def rxn_pred(self):
        self.reagent_mol_res = AllChem.DeleteSubstructs(self.reagent, self.mcs_rxn)

        self.pred_mol = AllChem.ReplaceSubstructs(self.mol, self.similar_mol_res, self.reagent_mol_res)[0]
        self.pred_smiles = Chem.MolToSmiles(self.pred_mol)
        self.pred_mol = Chem.MolFromSmiles(self.pred_smiles)
        self.pred_mol = Chem.RemoveHs(self.pred_mol)
        self.pred_smiles = Chem.MolToSmiles(self.pred_mol, canonical=True)

        if '.' not in self.pred_smiles:
            self.register_rxn()

    def register_rxn(self):
        self.pred_rxn = self.rxn.copy()

        self.database.register_mol(self.pred_smiles, self.rxn, self.mol_id)
        self.pred_id = self.database.mol_id

        if self.rxn.direct == -1:
            self.pred_rxn.react = self.mol_id
            self.pred_rxn['prod'] = self.pred_id

            self.pred_rxn.react_smiles = self.mol_smiles
            self.pred_rxn.prod_smiles = self.pred_smiles

        else:
            self.pred_rxn.react = self.pred_id
            self.pred_rxn['prod'] = self.mol_id

            self.pred_rxn.react_smiles = self.pred_smiles
            self.pred_rxn.prod_smiles = self.mol_smiles

        self.pred_rxn.enzyme_name = np.nan
        self.pred_rxn.reaction_formula = np.nan

        self.database.register_rxn(self.pred_rxn)


class PredictedDatabaseRegistration:
    def __init__(self):
        self.mols = load_kegg_mols()
        self.molDB = pd.DataFrame(columns=['SMILES', 'origin_rxn', 'origin_mol', 'direct'])
        self.mol_history = self.molDB.copy()

        self.rxns = load_single_kegg_rxns()
        self.rxnDB = pd.DataFrame(columns=self.rxns.columns)
        self.rxn_history = self.rxnDB.copy()

        self.mol_counter = 0
        self.rxn_counter = 0

    def register_mol(self, smiles, origin_rxn, origin_mol):
        # Check if the molecule in KEGG mols and molDB
        check_kegg = self.mols[self.mols.SMILES == smiles]
        check_db = self.molDB[self.molDB.SMILES == smiles]

        if not check_kegg.empty:
            self.mol_id = check_kegg.index[0]

        elif not check_db.empty:
            self.mol_id = check_db.index[0]

        else:
            self.mol_counter += 1
            self.mol_id = 'PC'+ str(self.mol_counter) # 'PC': Predicted compound
            self.molDB.loc[self.mol_id] = [smiles, origin_rxn.name, origin_mol, origin_rxn.direct]

            self.molDB.to_pickle(PRED_MOL_DB_PATH)

        self.temp_mol = pd.DataFrame(columns=['SMILES', 'origin_rxn', 'origin_mol', 'direct'])
        self.temp_mol.loc[self.mol_id] = [smiles, origin_rxn.name, origin_mol, origin_rxn.direct]
        self.mol_history = pd.concat([self.mol_history, self.temp_mol])

        self.mol_history.to_pickle(PRED_MOL_HISTORY_PATH)

    def register_rxn(self, rxn):
        # Check if the reaction in KEGG rxns and rxnDB
        def check_rxn(db, rxn):
            check_for = db.loc[(db.react_smiles == rxn.react_smiles) & (db.prod_smiles == rxn.prod_smiles)]
            if not check_for.empty:
                return check_for
            else:
                check_rev = db.loc[(db.react_smiles == rxn.prod_smiles) & (db.prod_smiles == rxn.react_smiles)]
                return check_rev

        check_kegg = check_rxn(self.rxns, rxn)
        check_db = check_rxn(self.rxnDB, rxn)

        if not check_kegg.empty:
            self.rxn_id = check_kegg.index[0]

        elif not check_db.empty:
            self.rxn_id = check_db.index[0]

        else:
            self.rxn_counter += 1
            self.rxn_id = 'PR'+ str(self.rxn_counter) # 'PC': Predicted reaction
            self.rxnDB.loc[self.rxn_id] = rxn

            self.rxnDB.to_pickle(PRED_RXN_DB_PATH)

        self.temp_rxn = pd.DataFrame(columns=self.rxns.columns)
        self.temp_rxn.loc[self.rxn_id] = rxn
        self.rxn_history = pd.concat([self.rxn_history, self.temp_rxn])

        self.rxn_history.to_pickle(PRED_RXN_HISTORY_PATH)


#%%
if __name__ == '__main':

    pred = MoleculePrediction()
    pred.iter_target_mols()