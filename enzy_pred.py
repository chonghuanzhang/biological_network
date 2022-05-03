#%%
import pandas as pd
import os

from utilsx import *
from cls_models import ProjectionModel, ImbalancedClassificaton
import itertools

from rdkit import Chem
from rdkit.Chem import AllChem

import pandas as pd
import numpy as np
from pylab import rcParams

from tensorflow.keras.utils import to_categorical

from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

# turn of system warning
import warnings
warnings.filterwarnings("ignore")
# turn of rdkit warning
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class EnzymePrediction:
    def __init__(self,
                 fp_type='bert',
                 model_method='projection',
                 radius=3,
                 nBits=128,
                 epoch=200,
                 batch=128,
                 lr=2e-3,
                 af='elu',
                 op='adam',
                 loss='binary_crossentropy',
                 ):

        self.nBits = nBits
        self.fp_type = fp_type
        self.model_method = model_method
        self.radius = radius
        self.nBits = nBits
        self.epoch = epoch
        self.batch = batch
        self.lr = lr
        self.af = af
        self.op = op
        self.loss = loss

        self.mols = load_kegg_mols()
        self.mols.dropna(subset=['SMILES'], inplace=True)

        self.rxns = load_kegg_rxns()

        self.single_rxns = load_single_kegg_rxns()
        self.single_rxns.dropna(subset=['react_smiles', 'prod_smiles'], inplace=True)

        self.enzys = load_enzymes()

        self.ec_token = load_ec_gene_token()
        self.ec_token.drop(['gene', 'seq'], axis=1, inplace=True)


    def main(self):

        self.find_promis_enzy()
        self.form_data()
        self.form_molecule_fps()
        self.form_rxn_fps()
        self.form_enzy_var()
        self.form_jaccard_dist()


    def execuate_train(self):
        if self.fp_type == 'ecfp':
            self.form_xys()

        elif self.fp_type == 'bert':
            self.form_bert_fps()
            self.form_xys_bert_fps()

        self.get_aaseq()

        if self.model_method == 'projection':
            self.train()

        elif self.model_method == 'imbalance':

            if self.fp_type == 'ecfp':
                self.x_ec_rxn()

            elif self.fp_type == 'bert':
                self.x_ec_rxn_bert_fps()

            self.imbalance_train()
            self.model.main()


    def form_bert_fps(self):
        def _rxn_fp(r,p):
            return r+'>>'+p

        self.data['react_smiles'] = list(self.data['react'].apply(lambda x: self.mols.loc[x].SMILES))
        self.data['prod_smiles'] = list(self.data['prod'].apply(lambda x: self.mols.loc[x].SMILES))

        self.data['rxn_smiles'] = list(self.data.apply(lambda x: _rxn_fp(x['react_smiles'], x['prod_smiles']), axis=1))

        model, tokenizer = get_default_model_and_tokenizer()
        rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
        fps = rxnfp_generator.convert_batch(self.data.rxn_smiles)

        self.bert_fps = pd.DataFrame(fps)
        self.bert_fps.index = self.data.index.get_level_values(1)
        self.bert_fps = self.bert_fps[~self.bert_fps.index.duplicated(keep='first')]


    def get_aaseq(self):
        tokens = self.ec_token.loc[self.Xs.index.get_level_values(0)]
        tokens.index = self.Xs.index
        self.Xs = pd.concat([self.Xs, tokens], axis=1)


    def find_promis_enzy(self):
        self.enzys = self.enzys.dropna(subset=['rxn'])
        self.promis_enzys = self.enzys.loc[self.enzys.rxn.map(len) > 1]


    def form_data(self):
        ECs = [i for i in self.enzys.index for rxn in self.enzys.loc[i].rxn if rxn in self.single_rxns.index]
        RXNs = [rxn for i in self.enzys.index for rxn in self.enzys.loc[i].rxn if rxn in self.single_rxns.index]

        self.data = pd.DataFrame(data={'react': self.single_rxns.loc[RXNs]['react'],
                                       'prod': self.single_rxns.loc[RXNs]['prod']},
                                 )
        self.data.index = pd.MultiIndex.from_tuples(list(zip(*[ECs, RXNs])), names=["EC", "RXN"])

        # some enzymes cannot find corresponding amino acid sequence. These enzymes are removed from data.
        excluded_ecs = set(ECs) - set(self.ec_token.index)
        self.data.drop(excluded_ecs, inplace=True)


    def x_ec_rxn_bert_fps(self):
        ys = []
        index = self.data.index
        ec = index.get_level_values(0).unique()
        rxn = index.get_level_values(1).unique()

        for i in itertools.product(ec, rxn):
            if i in index:
                ys.append(1)
            else:
                ys.append(0)

        all_data = pd.DataFrame(data={'y':ys},
                                index=pd.MultiIndex.from_tuples(list(itertools.product(ec, rxn)))
                                )

        all_data['react'] = list(self.single_rxns.loc[all_data.index.get_level_values(1)]['react'])
        all_data['y'] = ys

        all_data_one = all_data[all_data['y'] == 1]
        all_data_zero = all_data[all_data['y'] == 0]

        # The zero labelled rows consumes ~100 gb RAM. Delete 90%, while the zeros are still extremely dominated.
        np.random.seed(10)
        remove_n = int(len(all_data_zero)*0.9)
        drop_indices = np.random.choice(all_data_zero.index, remove_n, replace=False)
        all_data_zero.drop(drop_indices, inplace=True)

        all_data = pd.concat([all_data_one, all_data_zero], axis=0)


        rxn_fps = self.bert_fps.loc[all_data.index.get_level_values(1)]
        rxn_fps.index = all_data.index
        all_Xs = rxn_fps
        rxn_dist = self.rxn_dist[~self.rxn_dist.index.duplicated(keep='first')]
        all_Xs['dist'] = list(rxn_dist.loc[all_data.index.get_level_values(1)])

        ECs = self.ECs[0].loc[all_data.index.get_level_values(0)]
        expand_ecs = pd.DataFrame(to_categorical(ECs)).astype(int)
        expand_ecs.drop([0], axis=1, inplace=True)
        expand_ecs.index = all_Xs.index

        tokens = self.ec_token.loc[all_Xs.index.get_level_values(0)]
        tokens.index = all_Xs.index

        self.all_Xs = pd.concat([all_Xs, expand_ecs, tokens], axis=1, ignore_index=True)
        self.all_ys = all_data['y']


    def x_ec_rxn(self):
        ys = []
        index = self.data.index
        ec = index.get_level_values(0).unique()
        rxn = index.get_level_values(1).unique()

        for i in itertools.product(ec, rxn):
            if i in index:
                ys.append(1)
            else:
                ys.append(0)

        all_data = pd.DataFrame(data={'y':ys},
                                index=pd.MultiIndex.from_tuples(list(itertools.product(ec, rxn)))
                                )

        all_data['react'] = list(self.single_rxns.loc[all_data.index.get_level_values(1)]['react'])
        all_data['y'] = ys

        all_data_one = all_data[all_data['y'] == 1]
        all_data_zero = all_data[all_data['y'] == 0]

        # The zero labelled rows consumes ~100 gb RAM. Delete 90%, while the zeros are still extremely dominated.
        np.random.seed(10)
        remove_n = int(len(all_data_zero)*0.9)
        drop_indices = np.random.choice(all_data_zero.index, remove_n, replace=False)
        all_data_zero.drop(drop_indices, inplace=True)

        all_data = pd.concat([all_data_one, all_data_zero], axis=0)


        rxn_fps = self.rxn_fps.loc[all_data.index.get_level_values(1)]
        rxn_fps.index = all_data.index
        react_fps = self.mol_fps.loc[all_data.react]
        react_fps.index = all_data.index

        all_Xs = pd.concat([rxn_fps, react_fps],axis=1)
        rxn_dist = self.rxn_dist[~self.rxn_dist.index.duplicated(keep='first')]
        all_Xs['dist'] = list(rxn_dist.loc[all_data.index.get_level_values(1)])

        ECs = self.ECs[0].loc[all_data.index.get_level_values(0)]
        expand_ecs = pd.DataFrame(to_categorical(ECs)).astype(int)
        expand_ecs.drop([0], axis=1, inplace=True)
        expand_ecs.index = all_Xs.index

        tokens = self.ec_token.loc[all_Xs.index.get_level_values(0)]
        tokens.index = all_Xs.index

        self.all_Xs = pd.concat([all_Xs, expand_ecs, tokens], axis=1, ignore_index=True)
        self.all_ys = all_data['y']


    def form_rxn_fps(self):
        prods_fps = self.mol_fps.loc[self.single_rxns['prod']]
        prods_fps.index = self.single_rxns.index

        reacts_fps = self.mol_fps.loc[self.single_rxns['react']]
        reacts_fps.index = self.single_rxns.index

        self.rxn_fps = prods_fps - reacts_fps


    def form_molecule_fps(self):
        """convert from molecule id to morgan fingerprints"""
        def _m2fp(smi):
            rd_mol = Chem.MolFromSmiles(smi)
            return list(AllChem.GetMorganFingerprintAsBitVect(rd_mol, self.radius, nBits=self.nBits, useChirality=True))

        mols = self.mols.SMILES.apply(_m2fp)
        self.mol_fps = pd.DataFrame(list(mols), index=self.mols.index)


    def form_enzy_var(self):
        def expand_ec(ec_str):
            ec_str_lt = ec_str.split('.')
            ec_lt = [int(i) for i in ec_str_lt]
            return ec_lt

        self.ECs = pd.DataFrame(data=[expand_ec(ec) for ec in self.enzys.index],
                                index=self.enzys.index)


    def form_xys(self):
        rxn_fps = self.rxn_fps.loc[self.data.index.get_level_values(1)]
        rxn_fps.index = self.data.index
        react_fps = self.mol_fps.loc[self.data.react]
        react_fps.index = self.data.index

        self.Xs = pd.concat([rxn_fps, react_fps],axis=1)
        self.Xs['dist'] = list(self.rxn_dist)

        Ys = self.ECs.loc[self.data.index.get_level_values(0)]
        self.Ys = pd.DataFrame(to_categorical(Ys[0])).astype(int)
        self.Ys.drop([0], axis=1, inplace=True)


    def form_xys_bert_fps(self):
        rxn_fps = self.bert_fps.loc[self.data.index.get_level_values(1)]
        rxn_fps.index = self.data.index

        self.Xs = rxn_fps
        self.Xs['dist'] = list(self.rxn_dist)

        Ys = self.ECs.loc[self.data.index.get_level_values(0)]
        self.Ys = pd.DataFrame(to_categorical(Ys[0])).astype(int)
        self.Ys.drop([0], axis=1, inplace=True)


    def form_jaccard_dist(self):
        rxns = pd.DataFrame(index=self.data.index)
        rxns['react'] = list(self.mols.loc[self.data['react']].SMILES.apply(Chem.MolFromSmiles))
        rxns['prod'] = list(self.mols.loc[self.data['prod']].SMILES.apply(Chem.MolFromSmiles))

        self.rxn_dist = rxns.apply(lambda x: compute_jaccard(x['react'], x['prod']),axis=1)
        self.rxn_dist.index = self.rxn_dist.index.get_level_values(1)

    def train(self):
        self.model = ProjectionModel(optimizer=self.op,
                                     loss=self.loss,
                                     learning_rate=self.lr,
                                     epoch=self.epoch,
                                     batch_size=self.batch,
                                     activation_function=self.af,
                                     )

        self.model.X = self.Xs
        self.model.y = self.Ys

        # self.model.scale()
        self.model.fit()

    def imbalance_train(self):
        self.model = ImbalancedClassificaton(optimizer=self.op,
                                             loss=self.loss,
                                             learning_rate=self.lr,
                                             epoch=self.epoch,
                                             batch_size=self.batch,
                                             activation_function=self.af,
                                             )
        self.model.df = self.all_Xs
        self.model.df['y'] = self.all_ys


#%%
if __name__ == '__main__':

    """
    fp_type: 'ecfp', 'bert'
    model_method: 'projection', 'imbalance'
    """

    self = EnzymePrediction(fp_type='bert',
                            model_method='projection',
                            )
    self.main()
    self.execuate_train()