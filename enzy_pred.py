#%%
import pandas as pd
import os

from utilsx import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import multiprocessing as mp
import itertools
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from tensorflow.keras.utils import to_categorical


import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

import wandb
from wandb.keras import WandbCallback



class EnzymePrediction:
    def __init__(self,
                 nBits=128,
                 fp_type='bert',
                 model='projection'
                 ):
        self.nBits = nBits
        self.fp_type = fp_type
        self.model_method = model

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


    def form_bert_fps(self):
        def _rxn_fp(r,p):
            return r+'>>'+p

        self.data['react_smiles'] = list(self.data['react'].apply(lambda x: self.mols.loc[x].SMILES))
        self.data['prod_smiles'] = list(self.data['prod'].apply(lambda x: self.mols.loc[x].SMILES))

        rxn_smiles = []
        for index, row in self.data.iterrows():
            rxn_smiles.append(_rxn_fp(row.react_smiles, row.prod_smiles))

        model, tokenizer = get_default_model_and_tokenizer()

        rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

        fps = rxnfp_generator.convert_batch(rxn_smiles)
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
        self.model = ProjectionModel()

        self.model.X = self.Xs
        self.model.y = self.Ys

        # self.model.scale()
        self.model.fit()

    def imbalance_train(self):
        self.model = ImbalancedClassificaton()
        self.model.df = self.all_Xs
        self.model.df['y'] = self.all_ys


class ProjectionModel:
    def __init__(self):

        self.wandb = wandb

        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'
        self.metrics = ['accuracy']

        self.learning_rate = 0.001
        self.nb_epoch = 400
        self.batch_size = 64

    def compile_model(self):

        config = {
            "learning_rate": self.learning_rate,
            "epochs": self.nb_epoch,
            "batch_size": self.batch_size
        }

        self.wandb.init(project="synthetic_network_projection",
                        entity="chonghuanzhang",
                        config=config
                        )

        inputs = keras.Input(len(self.X.columns))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(50, activation='elu')(x)
        x = layers.Dropout(rate=0.6)(x)
        x = layers.Dense(30, activation='elu')(x)
        x = layers.Dropout(rate=0.4)(x)
        x = layers.Dense(10, activation='elu')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(5, activation='elu')(x)
        x = layers.Dropout(rate=0.2)(x)
        outputs = layers.Dense(7, activation='sigmoid')(x)

        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                self.model = keras.Model(inputs, outputs)
                self.model.compile(optimizer=self.optimizer,
                                   loss=self.loss,
                                   metrics=self.metrics)
        else:
            self.model = keras.Model(inputs, outputs)
            # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
            #                  loss='mse')  # , run_eagerly=True)

            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)

        # self.csv_logger = keras.callbacks.CSVLogger('log.csv', append=True, separator=';')

    def scale(self):
        self.transformer = MinMaxScaler().fit(self.y)

        # self.transformer = Normalizer().fit(self.y)
        # self.y_norm = self.transformer.transform(self.y)

        self.y_train_norm = self.transformer.transform(self.y_train)
        self.y_test_norm = self.transformer.transform(self.y_test)


    def fit(self):
        self.compile_model()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values,
                                                                                self.y.values,
                                                                                test_size=0.2)

        self.model.fit(self.X_train, self.y_train,
                       epochs=self.nb_epoch,
                       batch_size=self.batch_size,
                       validation_split=0.2,
                       shuffle=True,
                       verbose=1,
                       callbacks=[WandbCallback()])

        self.validate()


    def validate(self):
        self.model.evaluate(self.X_train, self.y_train, verbose=1)
        self.model.evaluate(self.X_test, self.y_test, verbose=1)

        self.y_trn_pred = self.model(self.X_train).numpy()
        self.y_trn_pred_class = np.argmax(self.y_trn_pred, axis=1)
        self.y_trn_class = np.argmax(self.y_train, axis=1)
        self.trn_result = pd.DataFrame(data={'trn': self.y_trn_class, 'trn_pred': self.y_trn_pred_class})

        self.y_test_pred = self.model(self.X_test).numpy()
        self.y_test_pred_class = np.argmax(self.y_test_pred, axis=1)
        self.y_test_class = np.argmax(self.y_test, axis=1)
        # 0 means class 1, 1 means class 2 ...
        self.test_result = pd.DataFrame(data={'test': self.y_test_class, 'test_pred': self.y_test_pred_class})
        len(self.test_result[self.test_result.test == self.test_result.test_pred])/len(self.test_result)


class ImbalancedClassificaton:
    """Autoencoder model for extreme imbalanced classification"""
    """Based on this page: https://towardsdatascience.com/extreme-rare-event-classification-using-autoencoders-in-keras-a565b386f098#:~:text=What%20is%20an%20extreme%20rare,%E2%80%9310%25%20of%20the%20total."""
    def __init__(self):

        self.wandb = wandb

        tf.random.set_seed(2)
        self.DATA_SPLIT_PCT = 0.2
        self.SEED = 123

        self.nb_epoch = 200
        self.batch_size = 128
        self.learning_rate = 2e-3

    def main(self):
        self.split_data()
        self.scale_data()
        self.compile_model()

    def split_data(self):
        self.df_train, self.df_test = train_test_split(self.df,
                                                       test_size=self.DATA_SPLIT_PCT,
                                                       random_state=self.SEED)
        self.df_train, self.df_valid = train_test_split(self.df_train,
                                                        test_size=self.DATA_SPLIT_PCT,
                                                        random_state=self.SEED)
        self.df_train_0 = self.df_train.loc[self.df['y'] == 0]
        self.df_train_1 = self.df_train.loc[self.df['y'] == 1]
        self.df_train_0_x = self.df_train_0.drop(['y'], axis=1)
        self.df_train_1_x = self.df_train_1.drop(['y'], axis=1)
        self.df_valid_0 = self.df_valid.loc[self.df['y'] == 0]
        self.df_valid_1 = self.df_valid.loc[self.df['y'] == 1]
        self.df_valid_0_x = self.df_valid_0.drop(['y'], axis=1)
        self.df_valid_1_x = self.df_valid_1.drop(['y'], axis=1)
        self.df_test_0 = self.df_test.loc[self.df['y'] == 0]
        self.df_test_1 = self.df_test.loc[self.df['y'] == 1]
        self.df_test_0_x = self.df_test_0.drop(['y'], axis=1)
        self.df_test_1_x = self.df_test_1.drop(['y'], axis=1)

    def scale_data(self):
        self.scaler = StandardScaler().fit(self.df_train_0_x)
        self.df_train_0_x_rescaled = self.scaler.transform(self.df_train_0_x)
        self.df_valid_0_x_rescaled = self.scaler.transform(self.df_valid_0_x)
        self.df_valid_x_rescaled = self.scaler.transform(self.df_valid.drop(['y'], axis=1))
        self.df_test_0_x_rescaled = self.scaler.transform(self.df_test_0_x)
        self.df_test_x_rescaled = self.scaler.transform(self.df_test.drop(['y'], axis=1))

    def compile_model(self):

        config = {
            "learning_rate": self.learning_rate,
            "epochs": self.nb_epoch,
            "batch_size": self.batch_size
        }

        self.wandb.init(project="synthetic_network_imbalance",
                        entity="chonghuanzhang",
                        config=config
                        )

        input_dim = self.df_train_0_x_rescaled.shape[1]  # num of predictor variables,

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(512,
                        activation="sigmoid",
                        activity_regularizer=regularizers.l1(self.learning_rate))(input_layer)
        encoder = Dense(256, activation="sigmoid")(encoder)
        encoder = Dense(128, activation="sigmoid")(encoder)
        encoder = Dense(64, activation="sigmoid")(encoder)
        decoder = Dense(128, activation="sigmoid")(encoder)
        decoder = Dense(256, activation="sigmoid")(decoder)
        decoder = Dense(512, activation="sigmoid")(decoder)
        decoder = Dense(input_dim, activation="linear")(decoder)
        self.autoencoder = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder.summary()

        self.autoencoder.compile(metrics=['accuracy'],
                                 loss='mean_squared_error',
                                 optimizer='adam')

        self.cp = ModelCheckpoint(filepath="autoencoder_classifier_20220315.h5",
                                  save_best_only=True,
                                  verbose=0)

        self.tb = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

        self.history = self.autoencoder.fit(self.df_train_0_x_rescaled, self.df_train_0_x_rescaled,
                                            epochs=self.nb_epoch,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            validation_data=(self.df_valid_0_x_rescaled, self.df_valid_0_x_rescaled),
                                            verbose=1,
                                            callbacks=[WandbCallback()]).history


    def select_threshold(self):
        self.valid_x_predictions = self.autoencoder.predict(self.df_valid_x_rescaled)
        mse = np.mean(np.power(self.df_valid_x_rescaled - self.valid_x_predictions, 2), axis=1)
        error_df = pd.DataFrame({'Reconstruction_error': mse,
                                'True_class': self.df_valid['y']})
        precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
        plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
        plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
        plt.title('Precision and recall for different threshold values')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()

    def test_data(self):
        self.test_x_predictions = self.autoencoder.predict(self.df_test_x_rescaled)
        mse = np.mean(np.power(self.df_test_x_rescaled - self.test_x_predictions, 2), axis=1)
        error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                                      'True_class': self.df_test['y']})
        error_df_test = error_df_test.reset_index()
        threshold_fixed = 0.4
        groups = error_df_test.groupby('True_class')
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                    label="Break" if name == 1 else "Normal")
        ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
        ax.legend()
        plt.title("Reconstruction error for different classes")
        plt.ylabel("Reconstruction error")
        plt.xlabel("Data point index")
        plt.show()



#%%
if __name__ == '__main__':

    """
    fp_type: 'ecfp', 'bert'
    model: 'projection', 'imbalance'
    """

    self = EnzymePrediction(fp_type='bert',
                            model='projection')
    self.main()
    self.execuate_train()