from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range

import pandas as pd
import numpy as np
import datetime
import os





def load_wearcodes(wearcodes_path):
    wearcodes = pd.read_csv(wearcodes_path)
    wearcodes['Day1'] = [datetime.datetime.strptime(s, '%d/%m/%Y') for s in
                         wearcodes['Day1']]
    wearcodes['Day2'] = [datetime.datetime.strptime(s, '%d/%m/%Y') for s in
                         wearcodes['Day2']]
    return wearcodes


def get_filenames(wearcodes):
    filenames = []
    for binfile in wearcodes['binFile']:
        for day in [1, 2]:
            fn = binfile + '_day' + str(day) + '.csv'
            filenames.append(fn)
    return filenames

def load_acceleromater_data(filename, tz='Europe/London'):
    data = pd.read_csv(filename,
                       index_col='timestamp', parse_dates=[0],
                       infer_datetime_format=True)
    data.index = data.index.tz_localize(tz)
    return data

def load_data(filename, tz='Europe/London'):
    """
    Loads the accelerometer data.
    The path to the file is: `[filepath]/[binfile]_[day].csv`
    Here, binfile refers to the binary accelerometer data data from which
    the 5-second accelerometer data (csv) was created.

    Parameters
    ----------
    filename : str
        complete path to the file

    Returns
    -------
    Pandas Dataframe with accelerometer data

    """
    result = None
    try:
        data = load_acceleromater_data(filename, tz=tz)
        data = data.dropna()
        data['filename'] = os.path.basename(filename)
        result = data
    except:
        print('File not found: ', filename)
        return None
    return result


def take_subsequences(dfs):
    """
    Make subsequences of the data that are completely valid.

    Parameters
    ----------
    dfs : dict
        dict holding all the merged dataframes (result from process_data)

    Returns
    -------
    dict holding all the subsequences

    """
    subsets = {}
    for key in list(dfs.keys()):
        dataset = dfs[key]
        invalids = [1] + list(dataset['invalid']) + [1]
        starts = [i for i in range(1, len(invalids) - 1) if invalids[i - 1] == 1 and invalids[i] == 0]
        ends = [i for i in range(1, len(invalids)) if invalids[i - 1] == 0 and invalids[i] == 1]
        dataset['subset'] = -1
        for i, (s, e) in enumerate(zip(starts, ends)):
            # Some minimum length
            if e - s > 300:
                dataset.loc[s - 1:e - 1, 'subset'] = i
                subsets[(key, i)] = (dataset[s - 1:e - 1].copy())
    return subsets


def save_subsequences(subsets, subsets_path):
    """
    Save the subsequences to the path: `[subsets_path]/[filename].csv`

    Parameters
    ----------
    subsets : dict
        Dict holding the dataframes of the subsequences
    subsets_path : str
        Directory to store the outputs
    """
    if not os.path.exists(subsets_path):
        os.makedirs(subsets_path)
    for fn, i in subsets:
        fn_out = str(str(i) + '_' +  fn)
        dat = subsets[(fn, i)]
        dat.to_csv(os.path.join(subsets_path, fn_out), date_format='%Y-%m-%dT%H:%M:%S%z')


def switch_positions(dfs):
    """
    Check the orientation position, and switch if necessary.
    The orientation is switched if the median of the x-angle is larger than 0.
    In this case, the x-angle and y-angle are mirrored.

    Parameters
    ----------
    dfs : dict
        Dict holding the dataframes of the sequences

    Returns
    -------
    Switched Dataframes

    """
    switch_columns = ['anglex', 'angley', 'roll_med_acc_x', 'roll_med_acc_y', 'dev_roll_med_acc_x',
                      'dev_roll_med_acc_y']
    for dataset in list(dfs.values()):
        dataset['switched_pos'] = False
        if 'heuristic' in dataset.columns:
            non_sleeping_indices = [x not in [1,2,7] for x in dataset['heuristic']]
            non_sleeping = dataset[non_sleeping_indices]
        else:
           non_sleeping = dataset
        if not non_sleeping.empty:
            # in the 'correct' orientation, anglex should be mostly negative
            med_x = np.median(non_sleeping['anglex'])
            if med_x > 0:
                for c in switch_columns:
                    if c in list(dataset.keys()):
                        dataset[c] *= -1
                dataset['switched_pos'] = True
                print('switched dataset with median %f'%med_x)
    return dfs




def process_data_onebyone(wearcodes, filepath, subsets_path):
    """
    Process the accelerometer files one-by-one.
    Creates subsequences, switches position
    and writes the data to the output path defined in argument subsets_path.

    Parameters
    ----------
    wearcodes_path : str
        path to wearcodes file with filenames
    filepath
    subsets_path
    """
    filenames = get_filenames(wearcodes)
    nr_files = 0
    nr_subsets = 0
    for filename in filenames:
        dfs = {}
        # Load data
        df = load_data(os.path.join(filepath, filename))
        if df is not None:
            nr_files += 1

            # Keep all data frames
            dfs[filename] = df

            # Take subsequences
            subsets = take_subsequences(dfs)
            nr_subsets += len(subsets)
            # Switch positions
            subsets_switched = switch_positions(subsets)

            # Save file
            save_subsequences(subsets_switched, subsets_path)
    print('{} out of {} files were read.'.format(nr_files, len(filenames)))
    print('{} subsets are stored in '.format(nr_subsets)+subsets_path)