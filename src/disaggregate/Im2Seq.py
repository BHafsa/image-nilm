# Author : Bousbiat Hafsa <hafsa.bousbiat@aau.at>

from __future__ import print_function, division
import math
from nilmtk.disaggregate import Disaggregator
import pandas as pd
import numpy as np
from collections import OrderedDict
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, GRU,Conv1D, Conv2D, Bidirectional, MaxPooling2D, GlobalAveragePooling2D, Flatten, \
    BatchNormalization, Dense, Dropout, Reshape, Concatenate, TimeDistributed
from pyts.image import GramianAngularField
from pyts.image import RecurrencePlot
from transformations.mtf import MarkovTransitionField
import random
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam, SGD

from disaggregate.batchgenerator import BatchGenerator

class ApplianceNotFoundError(Exception):
    pass



class Im2Seq(Disaggregator):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """
        self.img_method = params.get('img_method', 'gasf')
        self.MODEL_NAME = "Im2Seq_" + self.img_method
        self.models = OrderedDict()
        self.chunk_wise_training = params.get('chunk_wise_training', False)

        self.nb_cnn = params.get('nb_cnn')
        self.nb_dense = params.get('nb_dense')
        self.kernel_size = params.get('kernel_size')
        self.nb_filters = params.get('nb_filter')
        self.sequence_length = params.get('sequence_length')

        # self.sequence_length = params.get('sequence_length', 115)
        self.img_size = params.get('img_size', self.sequence_length)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size', 512)
        self.appliance_params = params.get('appliance_params', {})

        self.mains_min = 0
        self.mains_max = 32718

        self.retrain_params = params.get('retrain', None)

        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path', None)
        self.model_type = params.get('model_type', "simple")
        self.loss_object = tf.keras.losses.MeanAbsoluteError()
        if self.retrain_params:
            self.load_params()

        if self.load_model_path:
            self.load_model()

    def load_params(self):
        print("Loading pre-trained models")
        for appliance in self.retrain_params:
            model = self.return_network()
            model.load_weights(self.retrain_params[appliance])
            self.models[appliance] = model

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, **load_kwargs):
        """
                The partial fit function
        """
        # If no appliance wise parameters are specified, then they are computed from the data

        print("...............{} partial_fit running...............".format(self.MODEL_NAME))

        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)



        # TO preprocess the data and bring it to a valid shape
        train_main, train_appliances = self.call_preprocessing(train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)

        train_main = train_main.values.reshape((-1, self.sequence_length))

        new_train_appliances = []
        for app_name, app_dfs in train_appliances:
            app_df = pd.concat(app_dfs, axis=0)
            app_df_values = app_df.values.reshape((-1, self.sequence_length))
            new_train_appliances.append((app_name, app_df_values))

        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if len(train_main):
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = 'Im2Seq-dae-' + str(self.img_method) + '-' + str(self.sequence_length) + '-' + str(
                        self.model_type) + '-' + str(appliance_name) + '-temp-weights-18' + str(
                        random.randint(0, 100000)) + '.h5'
                    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                                 mode='min')
                    train_x, v_x, train_y, v_y = train_test_split(train_main, power, test_size=.15, random_state=10)

                    batch_generator = BatchGenerator(train_x, train_y, self.batch_size, self.img_method, self.img_size)
                    validation_batch_generator = BatchGenerator(v_x, v_y, self.batch_size, self.img_method,
                                                                self.img_size)

                    
                    
                    callbacks_list = [ checkpoint]

                    with tf.device('/device:GPU:0'):
                        print('training with GPU')
                        model.fit_generator(generator=batch_generator,
                                            steps_per_epoch=int(len(train_y) // self.batch_size),
                                            epochs=self.n_epochs,
                                            verbose=1,
                                            shuffle=True,
                                            callbacks=callbacks_list,
                                            validation_data=validation_batch_generator,
                                            validation_steps=int(len(v_y) // self.batch_size))
                    model.load_weights(filepath)

    def disaggregate_chunk(self, test_main_list, model=None, do_preprocessing=True):

        if model is not None:
            self.models = model
        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')
        test_predictions = []
        for test_mains_df in test_main_list:
            disggregation_dict = {}
            test_main_array = test_mains_df.values.reshape((-1, self.sequence_length))
            for appliance in self.models:
                prediction = []
                model = self.models[appliance]
                print(test_main_array.shape)

                batch_generator = BatchGenerator(test_main_array, None, self.batch_size, self.img_method, self.img_size)
                prediction = model.predict_generator(batch_generator)
                
                print('prediction finished')
                print(len(prediction))
                print(prediction.shape)
                #####################
                # This block is for creating the average of predictions over the different sequences
                # the counts_arr keeps the number of times a particular timestamp has occured
                # the sum_arr keeps the number of times a particular timestamp has occured
                # the predictions are summed for  agiven time, and is divided by the number of times it has occured
                l = self.sequence_length
                n = len(prediction) + l - 1
                sum_arr = np.zeros((n))
                counts_arr = np.zeros((n))
                o = len(sum_arr)
                for i in range(len(prediction)):
                    sum_arr[i:i + l] += prediction[i].flatten()
                    counts_arr[i:i + l] += 1
                for i in range(len(sum_arr)):
                    sum_arr[i] = sum_arr[i] / counts_arr[i]
                #################
                prediction = self.appliance_params[appliance]['mean'] + (
                        sum_arr * self.appliance_params[appliance]['std'])
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
                print(prediction.shape)
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def return_network(self):

        model = Sequential()
        # # defining the layers of the encoder
        model.add(Conv2D(filters=8, kernel_size=4, strides=2, activation='linear',
                         input_shape=(self.img_size, self.img_size, 1)))
        model.add(MaxPooling2D(strides=1, pool_size=(2, 2)))
        model.add(Dropout(.5))
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense((self.sequence_length) * 8, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense((self.sequence_length) * 8, activation='relu'))
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

        # # Hyperparameter of the Network

        sgd = tf.keras.optimizers.SGD(
            learning_rate=10e-3,
            momentum=.8
        )
        model.compile(loss="mse", optimizer=sgd)
        return model



    def call_preprocessing(self, mains_lst, submeters_lst, method):
        if method == 'train':
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_min) / (self.mains_max - self.mains_min)
                processed_mains_lst.append(pd.DataFrame(new_mains))
            # new_mains = pd.DataFrame(new_mains)
            appliance_list = []
            for app_index, (app_name, app_df_lst) in enumerate(submeters_lst):
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print("Parameters for ", app_name, " were not found!")
                    raise ApplianceNotFoundError()
                processed_app_dfs = []
                for app_df in app_df_lst:
                    new_app_readings = app_df.values.flatten()
                    new_app_readings = np.pad(new_app_readings, (units_to_pad, units_to_pad), 'constant',
                                              constant_values=(0, 0))
                    new_app_readings = np.array(
                        [new_app_readings[i:i + n] for i in range(len(new_app_readings) - n + 1)])
                    new_app_readings = (new_app_readings - app_mean) / app_std  # /self.max_val
                    processed_app_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_app_dfs))
                # new_app_readings = np.array([ new_app_readings[i:i+n] for i in range(len(new_app_readings)-n+1) ])
                # print (new_mains.shape, new_app_readings.shape, app_name)
            return processed_mains_lst, appliance_list

        else:
            processed_mains_lst = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                # new_mains = np.pad(new_mains, (units_to_pad,units_to_pad),'constant',constant_values = (0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_min) / (self.mains_max - self.mains_min)
                new_mains = new_mains.reshape((-1, self.sequence_length))
                processed_mains_lst.append(pd.DataFrame(new_mains))
            return processed_mains_lst

    def set_appliance_params(self, train_appliances):
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})