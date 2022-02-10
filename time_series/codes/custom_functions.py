import os

import numpy as np

import datetime

from tensorflow.data import Dataset
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import MeanSquaredError

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
            print(f'Make new directory {d}')
    return None


def mk_time_data(data):
    
    new_data = data.copy()

    new_data['Date'] = data['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    
    new_data['time_stamp'] = new_data['Date'].apply(lambda x: x.timestamp())
    
    new_data['year'] = new_data['Date'].apply(lambda x: x.year)
    new_data['month'] = new_data['Date'].apply(lambda x: x.month)
    new_data['day'] = new_data['Date'].apply(lambda x: x.day)
    
    time_stamp_tmp = new_data['time_stamp'].apply(lambda x: (x//(24*60*60))%365)
    new_data['cos_time_stamp'] = np.cos(2*np.pi*(time_stamp_tmp/365))
    new_data['sin_time_stamp'] = np.sin(2*np.pi*(time_stamp_tmp/365))
    
    return new_data


def mk_mean_std_dict(data):
    mean_std_dict = {
        col: {
            'mean': data[col].mean(),
            'std': data[col].std()
        } for col in data.columns
    }
    return mean_std_dict


def standard_scaling(data, mean_std_dict=None):
    if not mean_std_dict:
        mean_std_dict = mk_mean_std_dict(data)
    new_data = data.copy()
    for col in new_data.columns:
        new_data[col] -= mean_std_dict[col]['mean']
        new_data[col] /= mean_std_dict[col]['std']
    return new_data


def mk_dataset(data, CONFIGS, shuffle=False):
    
    time_series = data[CONFIGS['time_series_cols']][:-CONFIGS['target_length']]
    target_time_info = data[CONFIGS['target_time_info_cols']][CONFIGS['window_size']:]
    target = data[CONFIGS['target_cols']][CONFIGS['window_size']:]
    
    time_series_ds = Dataset.from_tensor_slices(time_series)
    time_series_ds = time_series_ds.window(CONFIGS['window_size'], shift=1, drop_remainder=True)
    time_series_ds = time_series_ds.flat_map(lambda x: x).batch(CONFIGS['window_size'])
    
    target_time_info_ds = Dataset.from_tensor_slices(target_time_info)
    
    target_ds = Dataset.from_tensor_slices(target)
    target_ds = target_ds.window(CONFIGS['target_length'], shift=1, drop_remainder=True)
    target_ds = target_ds.flat_map(lambda x: x).batch(CONFIGS['target_length'])
    
    ds = Dataset.zip(((time_series_ds, target_time_info_ds), target_ds))
    if shuffle:
        ds = ds.shuffle(CONFIGS['buffer_size'])
    ds = ds.batch(CONFIGS['batch_size']).cache().prefetch(2)
    
    return ds


def set_model(CONFIGS, model_name=None, print_summary=False):
    
    time_series_inputs = Input(batch_shape=(
        None, CONFIGS['window_size'], len(CONFIGS['time_series_cols'])
    ), name='time_series_inputs')
    
    conv_0 = Conv1D(16, 3, 2, activation='relu', name='conv_0')(time_series_inputs)
    pool_0 = MaxPool1D(2, name='pool_0')(conv_0)
    conv_1 = Conv1D(32, 3, 2, activation='relu', name='conv_1')(pool_0)
    pool_1 = MaxPool1D(2, name='pool_1')(conv_1)
    flatten = Flatten(name='flatten')(pool_1)
        
    target_time_info_inputs = Input(batch_shape=(
        None, len(CONFIGS['target_time_info_cols'])
    ), name='target_time_info_inputs')
    
    concat = Concatenate(name='concat')([flatten, target_time_info_inputs])
        
    dense_0 = Dense(64, activation='relu', name='dense_0')(concat)
    dense_1 = Dense(32, activation='relu', name='dense_1')(dense_0)
    outputs = Dense(len(CONFIGS['target_cols']), name='outputs')(dense_1)
    
    if not model_name:
        model_name = CONFIGS['model_name']
    
    model = Model(
        inputs = [time_series_inputs, target_time_info_inputs],
        outputs = outputs,
        name = model_name
    )
    
    optimizer = Adam(learning_rate=CONFIGS['learning_rate'])
    model.compile(
        loss = MeanSquaredError(),
        optimizer = optimizer,
    )
    
    if print_summary:
        model.summary()
    
    return model


def train_model(model, train_ds, valid_ds, CONFIGS):
    
    early_stop = EarlyStopping(
        patience=CONFIGS['es_patience']
    )
    save_best_only = ModelCheckpoint(
        filepath = f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5',
        monitor = 'val_loss',
        save_best_only = True,
        save_weights_only = True
    )
    tensorboard_callback = TensorBoard(
        log_dir = CONFIGS['tensorboard_log_path']
    )
    
    history = model.fit(
        train_ds,
        epochs = CONFIGS['epochs'],
        validation_data = valid_ds,
        callbacks = [
            early_stop,
            save_best_only,
            tensorboard_callback,
        ]
    )

    print('\n', '!'*30, '\n', '  Training is Done', '\n', '!'*30, sep='')
    
    return history
