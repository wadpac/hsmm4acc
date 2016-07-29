from __future__ import print_function

import pandas as pd
import dateutil.parser
import numpy as np
import datetime
import os


def loadData(name, filepath):
    result = {}
    for day in ['1', '2']:
        filename_csv = name + '_day' + day + '.csv'
        filename = filepath + filename_csv
        try:
            data = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
            data['filename'] = name + '_day' + day
            result[(name, day)] = data
        except:
            print('File not found: ', filename_csv)
            return None
    return result


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

    # Check if the differences between start and end is always 10 minutes
    differences = (annotations['end_time'] - annotations['start_time'])
    diff_indices = differences != datetime.timedelta(minutes=10)
    if sum(diff_indices) > 0:
        # TODO: this should be unnecessary!
        print('Differens between start and end not always 10 minutes! in %d cases.' % sum(diff_indices))
        # print(annotations[diff_indices])
        annotations.loc[diff_indices, 'end_time'] = datetime.timedelta(minutes=10) + annotations[diff_indices][
            'start_time']

    return annotations
