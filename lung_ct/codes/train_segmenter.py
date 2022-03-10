import os
import json
import argparse

import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import \
    EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from segmenter import *


class TrainSegmenter():

    def __init__(self, configs):
        self.configs = configs
        self.model_file_name = os.path.join(
            configs['model_path'],
            configs['model_name']+'.tf'
        )
        self.is_set_dataset = False

    @staticmethod
    def load_data(data_path, file_name):
        return np.load(os.path.join(data_path, file_name+'.npy'))

    @staticmethod
    def mk_dataset(X, y, shuffle=False, buffer_size=2048, batch_size=64):

        X_ds = Dataset.from_tensor_slices(X)
        y_ds = Dataset.from_tensor_slices(y)

        ds = Dataset.zip((X_ds, y_ds))
        if shuffle:
            ds = ds.shuffle(buffer_size)
        ds = ds.batch(batch_size).prefetch(2)

        return ds

    def set_dataset(self):
        self.train_ds = self.mk_dataset(
            self.load_data(self.configs['data_path'], 'X_train'),
            self.load_data(self.configs['data_path'], 'y_train'),
            shuffle=True,
            buffer_size=self.configs['buffer_size'],
            batch_size=self.configs['batch_size']
        )
        self.valid_ds = self.mk_dataset(
            self.load_data(self.configs['data_path'], 'X_valid'),
            self.load_data(self.configs['data_path'], 'y_valid'),
            buffer_size=self.configs['buffer_size'],
            batch_size=self.configs['batch_size']
        )
        self.is_set_dataset = True

    def set_model(self, print_summary=False):

        assert self.is_set_dataset, 'Set dataset before set model. self.set_dataset()'

        self.segmenter = Segmenter(self.configs['encoder_n_filters'],
                                   self.configs['decoder_n_filters'])
        loss = tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.configs['learning_rate'])
        self.segmenter.compile(optimizer=optimizer, loss=loss, metrics=['acc', 'mse'])

        sample = iter(self.train_ds).next()[0]
        _ = self.segmenter(sample)

        if print_summary:
            print(self.segmenter.summary())

    def train_model(self):

        early_stop = EarlyStopping(
            patience=self.configs['es_patience']
        )
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10,
            verbose=1, mode='auto', min_lr=1e-5
        )
        save_best_only = ModelCheckpoint(
            filepath = self.model_file_name,
            monitor = 'val_loss',
            save_best_only = True,
            save_weights_only = True
        )
        tensorboard_callback = TensorBoard(
            log_dir = self.configs['log_path']
        )

        history = self.segmenter.fit(
            self.train_ds,
            validation_data = self.valid_ds,
            epochs = self.configs['n_epochs'],
            callbacks = [
                early_stop,
                lr_scheduler,
                save_best_only,
                tensorboard_callback
            ]
        )
        self.segmenter.load_weights(self.model_file_name)


def get_args():
    
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--configs_file_name', type=str, default='../configs.json')

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--log_path', type=str)

    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=int)
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--es_patience', type=int)

    parser.add_argument('--encoder_n_filters', type=list)
    parser.add_argument('--decoder_n_filters', type=list)

    return parser.parse_args()


def main(args):

    CONFIGS_FILE_NAME = args.configs_file_name
    with open(CONFIGS_FILE_NAME) as f:
        configs = json.load(f)
    args_dict = vars(args)
    args_dict = {key: args_dict[key] for key in args_dict.keys() if args_dict[key] is not None}
    configs.update(args_dict)

    train_segmenter = TrainSegmenter(configs)
    train_segmenter.set_dataset()
    train_segmenter.set_model(print_summary=True)
    train_segmenter.train_model()


if __name__ == '__main__':

    args = get_args()
    if args is None:
        exit()

    main(args)
