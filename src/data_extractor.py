#!/usr/bin/env python3  

import wfdb
import os
import re
import numpy as np
from biosppy.signals import ecg

# Sorting of documents
data_files = os.listdir('../mit-bih-arrhythmia-database-1.0.0/')
data_files = np.array(data_files)
symbols = ['N','L','R','B','A','a','J','S','V', 'r','F','e','j','n','E','/','f','Q','?']

integers = [int(re.search(r'\d+', item).group()) for item in data_files if re.search(r'\d+', item)]
file_int = set(integers)
file_int = list(file_int)
file_int.remove(256)

padding = 300

for file in file_int:
    # The record contains all info about our experiment
    record = wfdb.rdsamp(f'../mit-bih-arrhythmia-database-1.0.0/{file}')

    # The label contains classification labels
    label = wfdb.rdann(f'../mit-bih-arrhythmia-database-1.0.0/{file}','atr')

    # transpose with data so I can have two lists
    data = record[0].transpose()

    # Set up our ground truth
    labels = np.array(label.symbol)
    gt = np.zeros(len(labels),dtype='float')
    for index, item in enumerate(labels):
        if item=='N':
            gt[index] = 1.0 #Normal
        elif item in symbols:
            gt[index] = 2.0 #Abnormal

    # For each channel we will feature extract
    # one row of our data is going to be a single pulse
    for index,channel in enumerate(data):

        # Channel's name 
        channel_name = record[1].get('sig_name')[index]

        # For more info on our experiment we'll use ecg lib
        out = ecg.ecg(signal=channel, sampling_rate=360, show=False)

        # Split channel data by each R peak of a pulse
        beats = np.split(channel, out['rpeaks'])

        # Write the data in csv form
        with open(f'../csv_files/{channel_name}/{channel_name}_{file}.csv','w') as f:
            print(channel_name)
            for i in range(padding):
                f.write(f'feature_{i + 1},')
            f.write('gt\n')
            for i,item in enumerate(beats):

                # If it's not classified don't take it into account
                if gt[i] == 0.0:
                    continue

                # If the pulse is super huge leave it out
                # The median number of samples in each pulse is about 200
                # So we'll use 200 features for each pulse
                if len(item) > padding:
                    continue

                # Zero padding
                zeros_to_pad = padding - len(item)
                item = np.pad(item, (0,padding - len(item)), 'constant', constant_values = 0.0)

                # Normalisation
                item = (item - item.min())/item.ptp()

                for jitem in item:
                    f.write(f'{jitem},')
                f.write(f'{gt[i] - 1.0}\n')

