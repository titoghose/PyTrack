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
        temp_dict['FixationSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['SaccadeSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['Blink'] = np.ones(len(temp_dict['Timestamp'])) * -1
        
        cnt = 0
        for e in d['events']['Efix']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['FixationSeq'][ind_start : ind_end + 1] = cnt
            cnt += 1

        cnt = 0
        for e in d['events']['Esac']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['SaccadeSeq'][ind_start : ind_end + 1] = cnt
            cnt += 1
        
        cnt = 0
        for e in d['events']['Eblk']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['Blink'][ind_start : ind_end + 1] = cnt
            cnt += 1

        df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
        del(temp_dict)

    df.to_csv(filename.split('.')[0] + '.csv')
    return df


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
        temp_dict['FixationSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['SaccadeSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['Blink'] = np.ones(len(temp_dict['Timestamp'])) * -1
        
        fix_cnt = 0
        sac_cnt = 0
        prev_end = 0
        for e in d['events']['Efix']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['FixationSeq'][ind_start : ind_end + 1] = fix_cnt
            fix_cnt += 1
            if prev_end != ind_start:
                temp_dict['SaccadeSeq'][prev_end + 1 : ind_start + 1] = sac_cnt
                sac_cnt += 1
            prev_end = ind_end

        cnt = 0
        for e in d['events']['Eblk']:
            ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
            ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
            temp_dict['Blink'][ind_start : ind_end + 1] = cnt
            cnt += 1

        df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
        del(temp_dict)

    df.to_csv(filename.split('.')[0] + '.csv') 
    return df


def csvToBase(filename):
    return


def convertToBase(filename):
    file_type = filename.split('.')[1]

    if file_type == 'asc':
        return eyeLinkToBase(filename)
    
    elif file_type == 'idf':
        return smiToBase(filename)

    elif file_type == 'csv':
        return csvToBase(filename)