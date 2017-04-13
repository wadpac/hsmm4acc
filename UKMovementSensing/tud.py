from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from past.utils import old_div

import pandas as pd
import dateutil.parser
import numpy as np
import datetime
import os

def parse_time(date_string):
    try:
        return pd.to_datetime(date_string, utc=True).tz_convert('Europe/London')
    except:
        print('Could not parse: ', date_string)
        return None


def process_annotations(annotations_path):
    """
    Read the time use diary (TUD) annotations file as stored in csv format
    Parameters
    ----------
    annotations_path : str
        Path to TUD file

    Returns
    -------
    Pandas dataframe with annotations

    """
    annotations = pd.read_csv(annotations_path)

    # Convert timestamps to datetime
    annotations['start_time'] = [parse_time(s) for s in annotations['start_time']]
    annotations['end_time'] = [parse_time(s) for s in annotations['end_time']]

    # Sort on measure day and time slot
    annotations = annotations.sort_values(['accSmallID', 'day', 'slot'])
    annotations.index = list(range(annotations.shape[0]))

    # Check if the differences between start and end is always 10 minutes
    differences = [x - y for x,y in zip(annotations['end_time'], annotations['start_time'])]
    diff_indices = [x != pd.Timedelta(10, unit='m') for x in differences]
    if sum(diff_indices) > 0:
        # TODO: this should be unnecessary!
        print('Differens between start and end not always 10 minutes! in %d cases. First 10:' % sum(diff_indices))
        print(annotations[diff_indices][['start_time', 'end_time']].head(10))
        annotations.loc[diff_indices, 'end_time'] = [pd.Timedelta(10, unit='m') + x for x in annotations[diff_indices]['start_time']]

    # Check if every timeslot is associated with exactly one time
    annotations['end_time_time'] = [s.time() for s in annotations['end_time']]
    byslot_endtime = annotations.groupby(['slot']).end_time_time
    multiple_endtime = (byslot_endtime.nunique() > 1)
    if sum(multiple_endtime)>0:
        print('Multiple end times per slot. first 10:')
        print(byslot_endtime.unique()[multiple_endtime])
    annotations.drop('end_time_time', 1, inplace=True)

    # Check if each day has exactly 144 time slots
    nrslots = annotations.groupby(['accSmallID', 'day']).slot.nunique()
    if not np.alltrue(nrslots == 144):
        print('Not all days have 144 slots! first 5:')
        print(nrslots[nrslots!=144].head())

    starttimes = annotations.groupby(['day', 'accSmallID'])['start_time']
    doubles = starttimes.nunique() < starttimes.count()
    if (sum(doubles) > 0):
        print('{} id/day combination with double start_time, first 5:'
              .format(sum(doubles)))
        print(starttimes.count()[doubles].head(5))

    return annotations


def join_wearcodes(wearcodes, annotations, mergeID = 'accSmallID'):
    """
    Joins the annotations with the wearcodes. The wearcodes are the dates on
    which an accelerometer was supposed to be worn

    Parameters
    ----------
    wearcodes_path : str
        path to the file with wearcodes
    annotations : pandas dataframe
        Dataframe with the annotations (output from process_annotations)

    Returns
    -------
    Pandas Dataframe with annotations and wearcodes

    """

    # Read in the wearcodes


    # Join with annotations
    annotations_codes = pd.merge(annotations, wearcodes, on=mergeID, how='left')

    # Check if all wearcodes are present
    if(sum(annotations_codes['Monitor'] == None) > 0):
        print('Some {} are not present! First 5:'.format(mergeID))
        print(annotations_codes[annotations_codes['Monitor'] == None].head())

    return annotations_codes


def add_annotations(df, annotations_group):
    """
    Add annotations to accelerometer data.

    Parameters
    ----------
    df : Pandas DataFrame
        Accelerometer data
    annotations_group
        Annotations for this specific binfile and day

    Returns
    -------
    Pandas DataFrame with accelerometer data and annotations.
    """
    # Add rounded time
    df['start_time'] = [tm - datetime.timedelta(minutes=tm.minute % 10,
                                                seconds=tm.second,
                                                microseconds=tm.microsecond)
                        for tm in df.index]

    df_annotated = pd.merge(df, annotations_group[
        ['activity', 'label', 'start_time', 'slot']], on='start_time', how='left')
    df_annotated.index = df.index
    return df_annotated