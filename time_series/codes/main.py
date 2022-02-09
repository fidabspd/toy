VER = 0

from custom_functions import *
import argparse
import pandas as pd


def parse_args():
    desc = "SET CONFIGS"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--file_name', type=str, help='**.csv')

    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--model_path', type=str, default='../model/')
    parser.add_argument('--model_name', type=str, default='time_series')

    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--es_patience', type=int, default=10)

    parser.add_argument('--target_seq_len', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=7*10)
    # parser.add_argument('--shift', type=int, default=)
    # parser.add_argument('--target_length', type=str, default=)

    return parser.parse_args()


def main(args):

    CONFIGS = vars(args)
    CONFIGS['shift'] = 1
    CONFIGS['target_length'] = 1
    CONFIGS['tensorboard_log_path'] = f'../logs/tensorboard/{CONFIGS["model_name"]}_ver{VER}'
    CONFIGS['time_series_cols'] = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'time_stamp',
    ]
    CONFIGS['target_time_info_cols'] = [
        'time_stamp', 'cos_time_stamp', 'sin_time_stamp',
    ]
    CONFIGS['target_cols'] = [
        'Open', 'High', 'Low', 'Close',
    ]

    scaling_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'time_stamp',
    ]

    data = pd.read_csv(CONFIGS['data_path']+CONFIGS['file_name'])
    data = mk_time_data(data)
    mean_std_dict = mk_mean_std_dict(data[:-CONFIGS['target_seq_len']*2][scaling_cols])
    data[scaling_cols] = standard_scaling(data[scaling_cols], mean_std_dict)

    train = data[:-CONFIGS['target_seq_len']*2]
    valid = data[-CONFIGS['target_seq_len']*2-CONFIGS['window_size']:-CONFIGS['target_seq_len']]
    test = data[-CONFIGS['target_seq_len']-CONFIGS['window_size']:]

    train_ds = mk_dataset(train, CONFIGS, shuffle=True)
    valid_ds = mk_dataset(valid, CONFIGS)
    test_ds = mk_dataset(test, CONFIGS)

    model = set_model(CONFIGS, print_summary=True)

    history = train_model(model, train_ds, valid_ds, CONFIGS)

    best_model = set_model(CONFIGS, model_name='best_'+CONFIGS['model_name'])
    best_model.load_weights(f'{CONFIGS["model_path"]}{CONFIGS["model_name"]}.h5')

    train_loss = best_model.evaluate(train_ds, verbose=0)
    valid_loss = best_model.evaluate(valid_ds, verbose=0)
    test_loss = best_model.evaluate(test_ds, verbose=0)

    print('\n', '='*30, sep='')
    print(' '*12, 'RESULT', '\n', sep='')
    print(f'  train_loss: {train_loss:.6f}')
    print(f'  valid_loss: {valid_loss:.6f}')
    print(f'  test_loss:  {test_loss:.6f}')
    print('='*30, '\n')

    # y_train_pred = best_model.predict(train_ds)
    # y_valid_pred = best_model.predict(valid_ds)
    # y_test_pred = best_model.predict(test_ds)


if __name__ == '__main__':

    args = parse_args()
    if args is None:
        exit()

    dirs = [
        '../logs/',
        '../logs/tensorboard/',
        '../model/',
    ]
    check_dirs(dirs)

    main(args)
