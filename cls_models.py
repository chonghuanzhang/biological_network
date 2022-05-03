from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

import multiprocessing as mp
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import wandb
from wandb.keras import WandbCallback


class ProjectionModel:
    def __init__(self,
                 optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'],
                 learning_rate=1e-3,
                 epoch=400,
                 batch_size=64,
                 activation_function='elu',
                 ):

        self.wandb = wandb

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        self.learning_rate = learning_rate
        self.nb_epoch = epoch
        self.batch_size = batch_size
        self.acti_fc = activation_function

    def compile_model(self):

        config = {
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
            "learning_rate": self.learning_rate,
            "epochs": self.nb_epoch,
            "batch_size": self.batch_size,
            'actiion_function': self.acti_fc
        }

        self.wandb.init(project="synthetic_network_projection",
                        entity="chonghuanzhang",
                        config=config
                        )

        inputs = keras.Input(len(self.X.columns))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dense(50, activation=self.acti_fc)(x)
        x = layers.Dropout(rate=0.6)(x)
        x = layers.Dense(30, activation=self.acti_fc)(x)
        x = layers.Dropout(rate=0.4)(x)
        x = layers.Dense(10, activation=self.acti_fc)(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(5, activation=self.acti_fc)(x)
        x = layers.Dropout(rate=0.2)(x)
        outputs = layers.Dense(7, activation='linear')(x)

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
    def __init__(self,
                 optimizer='adam',
                 loss='mean_squared_error',
                 learning_rate='2e-3',
                 metrics=['accuracy'],
                 epoch=100,
                 batch_size=64,
                 activation_function='elu'
                 ):

        self.wandb = wandb

        tf.random.set_seed(2)
        self.DATA_SPLIT_PCT = 0.2
        self.SEED = 123

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.nb_epoch = epoch
        self.batch_size = batch_size
        self.acti_fc = activation_function

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
            "optimizer": self.optimizer,
            "loss": self.loss,
            "metrics": self.metrics,
            "learning_rate": self.learning_rate,
            "epochs": self.nb_epoch,
            "batch_size": self.batch_size,
            'actiion_function': self.acti_fc,
        }

        self.wandb.init(project="synthetic_network_imbalance",
                        entity="chonghuanzhang",
                        config=config
                        )

        input_dim = self.df_train_0_x_rescaled.shape[1]  # num of predictor variables,

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(512,
                        activation=self.acti_fc,
                        activity_regularizer=regularizers.l1(self.learning_rate))(input_layer)
        encoder = Dense(512, activation=self.acti_fc)(encoder)
        encoder = Dense(256, activation=self.acti_fc)(encoder)
        encoder = Dense(256, activation=self.acti_fc)(encoder)
        encoder = Dense(128, activation=self.acti_fc)(encoder)
        encoder = Dense(128, activation=self.acti_fc)(encoder)
        encoder = Dense(64, activation=self.acti_fc)(encoder)
        encoder = Dense(64, activation=self.acti_fc)(encoder)
        decoder = Dense(128, activation=self.acti_fc)(encoder)
        decoder = Dense(128, activation=self.acti_fc)(decoder)
        decoder = Dense(256, activation=self.acti_fc)(decoder)
        decoder = Dense(256, activation=self.acti_fc)(decoder)
        decoder = Dense(512, activation=self.acti_fc)(decoder)
        decoder = Dense(512, activation=self.acti_fc)(decoder)
        decoder = Dense(input_dim, activation="linear")(decoder)
        self.autoencoder = Model(inputs=input_layer, outputs=decoder)
        self.autoencoder.summary()

        self.autoencoder.compile(metrics=self.metrics,
                                 loss=self.loss,
                                 optimizer=self.optimizer)

        self.cp = ModelCheckpoint(filepath="autoencoder_classifier.h5",
                                  save_best_only=True,
                                  verbose=1)

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

