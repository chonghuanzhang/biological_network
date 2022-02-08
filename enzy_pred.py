#%%
import pandas as pd
import os

path = '/home/cz332/programming/aaseqCrawler'
from utilsx import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import multiprocessing as mp
import itertools
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


import warnings
warnings.filterwarnings("ignore")

class EnzymePrediction:
    def __init__(self,
                 nBits=1024):
        self.nBits = nBits

        self.mols = load_kegg_mols()
        self.mols.dropna(subset=['SMILES'], inplace=True)

        self.rxns = load_kegg_rxns()

        self.single_rxns = load_single_kegg_rxns()
        self.single_rxns.dropna(subset=['react_smiles', 'prod_smiles'], inplace=True)

        self.enzys = load_enzymes()


    def main(self):
        self.find_promis_enzy()
        self.form_data()
        self.form_molecule_fps()
        self.form_rxn_fps()

        self.form_enzy_var()
        self.form_xys()

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

    def form_all_data(self):
        ys = []
        index = self.data.index
        for i in itertools.product(self.enzys.index, self.single_rxns.index):
            if i in index:
                ys.append(1)
            else:
                ys.append(0)

        all_data = pd.DataFrame(data={'y':ys},
                                index=pd.MultiIndex.from_tuples(list(itertools.product(self.enzys.index, self.single_rxns.index)))
                                )

        all_data_forward = all_data.copy()
        all_data_reverse = all_data.copy()
        all_data_forward['react'] = list(self.single_rxns.loc[all_data.index.get_level_values(1)]['react'])
        # all_data_forward['prod'] = list(self.single_rxns.loc[all_data.index.get_level_values(1)]['prod'])
        all_data_reverse['react'] = list(self.single_rxns.loc[all_data.index.get_level_values(1)]['prod'])
        #　all_data_reverse['prod'] = list(self.single_rxns.loc[all_data.index.get_level_values(1)]['react'])

        self.all_data = pd.concat([all_data_forward, all_data_reverse], axis=0)

    def form_rxn_fps(self):
        prods_fps = self.mol_fps.loc[self.single_rxns['prod']]
        prods_fps.index = self.single_rxns.index

        reacts_fps = self.mol_fps.loc[self.single_rxns['react']]
        reacts_fps.index = self.single_rxns.index

        self.rxn_fps = prods_fps - reacts_fps

    def form_molecule_fps(self):

        def m2fp(mol):
            """convert from molecule id to morgan fingerprints"""
            rd_mol = Chem.MolFromSmiles(mol)
            return list(AllChem.GetMorganFingerprintAsBitVect(rd_mol, 2, nBits=self.nBits, useChirality=True))

        mols = self.mols.SMILES.apply(m2fp)
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
        react_fps = self.mol_fps.loc[self.data.react]
        react_fps.index = self.data.index.get_level_values(1)

        self.Xs = pd.concat([rxn_fps, react_fps],axis=1)
        self.Ys = self.ECs.loc[self.data.index.get_level_values(0)]


    def train(self):
        self.model = ProjectionModel()
        self.model.nBits = self.nBits

        self.model.X = self.Xs
        self.model.y = self.Ys

        self.model.fit()


class ProjectionModel:
    def __init__(self):
        pass

    def compile_model(self):
        inputs = keras.Input(shape=(self.nBits*2))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(10, activation='relu')(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Dense(7, activation='relu')(x)
        x = layers.Dropout(rate=0.1)(x)
        outputs = layers.Dense(4, activation='relu')(x)
        # outputs = (1 - keras.activations.exponential(-tf.abs(x)))*500

        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = keras.Model(inputs, outputs)
                self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                                   loss='mse')  # , run_eagerly=True)
        else:
            self.model = keras.Model(inputs, outputs)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                               loss='mse')  # , run_eagerly=True)

        # self.csv_logger = keras.callbacks.CSVLogger('log.csv', append=True, separator=';')

    def fit(self):
        self.compile_model()

        self.transformer = Normalizer().fit(self.y)
        self.y_norm = self.transformer.transform(self.y)


        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values,
                                                                                self.y_norm,
                                                                                test_size=0.2)
        self.model.fit(self.X_train, self.y_train,
                       batch_size=64, epochs=100, validation_split=0.2,
                       )
        self.validate()

    def validate(self):
        self.y_trn_pred = self.model(self.X_train).numpy()
        self.trn_mse = keras.losses.MeanSquaredError()(self.y_trn_pred, self.y_train).numpy()
        self.scaled_trn_rmse = np.sqrt(self.trn_mse) / self.y_train.mean()
        logger.info('training data mse: {}, scaled_rmse: {}'.format(self.trn_mse, self.scaled_trn_rmse))

        self.y_pred = self.model(self.X_test).numpy()
        self.mse = keras.losses.MeanSquaredError()(self.y_test, self.y_pred).numpy()
        self.scaled_rmse = np.sqrt(self.mse) / self.y_test.mean()
        logger.info('test data mse: {}, scaled_rmse: {}'.format(self.mse, self.scaled_rmse))



#%%
self = EnzymePrediction(nBits=56)
self.main()
self.train()