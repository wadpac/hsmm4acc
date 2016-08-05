from __future__ import print_function

import pandas as pd
import dateutil.parser
import numpy as np
import datetime
import os


def parse_time(date_string):
    try:
        return dateutil.parser.parse(date_string)
    except:
        print('Could not parse: ', date_string)
        return None


def remove_invalid_annotations(annotations):
    # TODO: we can remove this function if we no longer have missing dates
    # Remove entries with missing start or endtime
    starttime_missing = annotations['start_time'].isnull()
    endtime_missing = annotations['end_time'].isnull()
    print('Number of missing start and end times:', sum(starttime_missing), sum(endtime_missing))
    keep = np.logical_not(np.logical_or(starttime_missing, endtime_missing))
    annotations_valid = annotations[keep]
    return annotations_valid


def process_annotations(annotations_path):
    annotations = pd.read_csv(annotations_path)
    annotations = remove_invalid_annotations(annotations)

    # Convert timestamps to datetime
    annotations['start_time'] = [parse_time(s) for s in annotations['start_time']]
    annotations['end_time'] = [parse_time(s) for s in annotations['end_time']]

    #sort on measure day and time slot
    annotations = annotations.sort_values(['serflag', 'tud_day', 'Slot'])
    annotations.index = range(annotations.shape[0])

    # Check if the differences between start and end is always 10 minutes
    differences = (annotations['end_time'] - annotations['start_time'])
    diff_indices = differences != datetime.timedelta(minutes=10)
    if sum(diff_indices) > 0:
        # TODO: this should be unnecessary!
        print('Differens between start and end not always 10 minutes! in %d cases. First 10:' % sum(diff_indices))
        print(annotations[diff_indices][['start_time', 'end_time']].head(10))
        annotations.loc[diff_indices, 'end_time'] = datetime.timedelta(minutes=10) + annotations[diff_indices]['start_time']

    # Check if every timeslot is associated with exactly one time
    annotations['end_time_time'] = [s.time() for s in annotations['end_time']]
    byslot_endtime = annotations.groupby(['Slot']).end_time_time
    multiple_endtime = (byslot_endtime.nunique() > 1)
    if sum(multiple_endtime)>0:
        print('Multiple end times per slot. first 10:')
        print(byslot_endtime.unique()[multiple_endtime.index].head(10))
    annotations.drop('end_time_time', 1, inplace=True)

    #Check if each day has exactly 144 time slots
    nrslots = annotations.groupby(['serflag', 'tud_day']).Slot.nunique()
    if not np.alltrue(nrslots == 144):
        print('Not all days have 144 slots! first 5:')
        print(nrslots[nrslots!=144].head())


    return annotations

def join_wearcodes(wearcodes_path, annotations):
    # Read in the wearcodes
    wearcodes = pd.read_csv(wearcodes_path)
    wearcodes['Day1'] = [datetime.datetime.strptime(s, '%d/%m/%Y') for s in wearcodes['Day1']]
    wearcodes['Day2'] = [datetime.datetime.strptime(s, '%d/%m/%Y') for s in wearcodes['Day2']]

    # Join with annotations
    annotations_codes = pd.merge(annotations, wearcodes, on='serflag', how='left')

    # Check if all wearcodes are present
    if(sum(annotations_codes['Monitor'] == None) > 0):
        print('Some serflags are not present! First 5:')
        print(annotations_codes[annotations_codes['Monitor'] == None].head())

    return annotations_codes

def load_data(binfile, day, filepath):
    result = None
    filename_csv = binfile + '_day' + str(day) + '.csv'
    filename = filepath + filename_csv
    try:
        data = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
        data = data.dropna()
        data['filename'] = binfile + '_day' + str(day)
        result = data
    except:
        print('File not found: ', filename_csv)
        return None
    return result

def add_annotations(df, binfile, day, annotations_group):
    firsttime = df.index[0]
    if firsttime.tz_localize('UTC') != min(annotations_group.start_time) :
        print('starttime of data does not correspond with starttime of annotations!')
        print(firsttime, min(annotations_group.start_time))
    # Add slot column to df
    df['Slot'] = df.index - firsttime
    df['Slot'] = [int(np.floor(s.total_seconds() / 600)) + 1 for s in df['Slot']]
    if(len(df.Slot.unique())<144):
        print('Warning: only %d slots'%len(df.Slot.unique()))
    df_annotated = pd.merge(df, annotations_group[['Slot', 'act', 'act_label', 'start_time']], on='Slot', how='left')
    df_annotated.index = df.index
    return df_annotated

def process_data(annotations_codes, filepath):
    byName = annotations_codes.groupby(['binFile', 'tud_day'])
    dfs = {}
    for (binfile, day), fileAnnotations in byName:
        annotations_group = byName.get_group((binfile, day))
        # Load data
        df = load_data(binfile, day, filepath)

        if df is not None:
            # Add annotations:
            df = add_annotations(df, binfile, day, annotations_group)
            # Keep all data frames
            dfs[(binfile, day)] = df
    return dfs

if __name__ == "__main__":
    import sys
    file_paths = sys.argv[1]
    annotations_path = sys.argv[2]
    wearcodes_path = sys.argv[3]
    annotations = process_annotations(annotations_path)
    annotations_codes = join_wearcodes(wearcodes_path, annotations)