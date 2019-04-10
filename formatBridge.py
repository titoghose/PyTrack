import subprocess
from pygazeanalyser.edfreader import read_edf
from pygazeanalyser.idfreader import read_idf
import pandas as pd
import numpy as np
import sys

col_headers = ['Timestamp', 'StimulusName', 'EventSource', 'GazeLeftx', 'GazeRightx', 'GazeLefty', 'GazeRighty', 'PupilLeft', 'PupilRight', 'FixationSeq', 'SaccadeSeq', 'Blink']

def eyeLinkToBase(filename):
    global col_headers

    data = read_edf(filename, 'START', missing=-1.0)
    df = pd.DataFrame(columns=col_headers)

    for d in data:
        temp_dict = dict.fromkeys(col_headers)

        temp_dict['Timestamp'] = d['trackertime']
        temp_dict['StimulusName'] = ['stim_name'] * len(temp_dict['Timestamp'])
        temp_dict['EventSource'] = ['ET'] * len(temp_dict['Timestamp'])
        temp_dict['GazeLeftx'] = d['x']
        temp_dict['GazeRightx'] = d['x']
        temp_dict['GazeLefty'] = d['y']
        temp_dict['GazeRighty'] = d['y']
        temp_dict['PupilLeft'] = d['size']
        temp_dict['PupilRight'] = d['size']
        temp_dict['FixationSeq'] = np.zeros(len(temp_dict['Timestamp']))
        temp_dict['SaccadeSeq'] = np.zeros(len(temp_dict['Timestamp']))
        temp_dict['Blink'] = np.zeros(len(temp_dict['Timestamp']))
        
        for e in d['events']['Efix']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['FixationSeq'][ind_start : ind_end + 1] = 1

        for e in d['events']['Esac']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['SaccadeSeq'][ind_start : ind_end + 1] = 1
        
        for e in d['events']['Eblk']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['Blink'][ind_start : ind_end + 1] = 1

        df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
        del(temp_dict)

    df.to_csv(filename.split('.')[0] + '.csv') 


def smiToBase(filename):
    global col_headers
    
    data = read_idf(filename, 'START', missing=-1.0)
    df = pd.DataFrame(columns=col_headers)

    for d in data:
        temp_dict = dict.fromkeys(col_headers)

        temp_dict['Timestamp'] = d['trackertime']
        temp_dict['StimulusName'] = ['stim_name'] * len(temp_dict['Timestamp'])
        temp_dict['EventSource'] = ['ET'] * len(temp_dict['Timestamp'])
        temp_dict['GazeLeftx'] = d['x']
        temp_dict['GazeRightx'] = d['x']
        temp_dict['GazeLefty'] = d['y']
        temp_dict['GazeRighty'] = d['y']
        temp_dict['PupilLeft'] = d['size']
        temp_dict['PupilRight'] = d['size']
        temp_dict['FixationSeq'] = np.zeros(len(temp_dict['Timestamp']))
        temp_dict['SaccadeSeq'] = np.zeros(len(temp_dict['Timestamp']))
        temp_dict['Blink'] = np.zeros(len(temp_dict['Timestamp']))
        

        for e in d['events']['Efix']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['FixationSeq'][ind_start : ind_end + 1] = 1

        temp_dict['SaccadeSeq'][np.where(temp_dict['FixationSeq'] == 0)[0]] = 1
        
        for e in d['events']['Eblk']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['Blink'][ind_start : ind_end + 1] = 1

        df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
        del(temp_dict)

    df.to_csv(filename.split('.')[0] + '.csv') 


def csvToBase(filename):
    return


def convertToBase(filename):
    file_type = filename.split('.')[1]

    if file_type == 'asc':
        eyeLinkToBase(filename)
    
    elif file_type == 'idf':
        smiToBase(filename)

    elif file_type == 'csv':
        csvToBase(filename)