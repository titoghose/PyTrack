import subprocess
from pygazeanalyser.edfreader import read_edf
# from pygazeanalyser.idfreader import read_idf
import pickle
import scipy.io as sio
import pandas as pd
import numpy as np
import json
import sys
import mne

col_headers = ['Timestamp', 'StimulusName', 'EventSource', 'GazeLeftx', 'GazeRightx', 'GazeLefty', 'GazeRighty', 'PupilLeft', 'PupilRight', 'FixationSeq', 'SaccadeSeq', 'Blink']

def eyeLinkToBase(filename, sensor_info):
    """
    Bridge function that converts ASCII (asc) EyeLink eye tracking data files to the base CSV format that the framework uses.

    Parameters
    ----------
    filename : str 
        Full file name (with path) of the data file
    
    sensor_info : str | dict
        If str, it is the name of the json file containing experiment details. If dict, contains the details of the sensor parameters as a dictionary of values (see documentation for details on needed parameters).

    Returns
    -------
    df : pandas DataFrame  
        Pandas dataframe of the data in the framework friendly base csv format 
    """
    global col_headers

    data = read_edf(filename, 'START', missing=-1.0)
    df = pd.DataFrame(columns=col_headers)

    i = 0
    for d in data:
        temp_dict = dict.fromkeys(col_headers)

        temp_dict['Timestamp'] = d['trackertime']
        temp_dict['StimulusName'] = ['stimulus_' + str(i)] * len(temp_dict['Timestamp'])
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
        
        i += 1

    df.to_csv(filename.split('.')[0] + '.csv')
    return df


def smiToBase(filename, sensor_info):
    """
    Bridge function that converts SMI (idf) raw eye tracking data files to the base CSV format that the framework uses.

    Parameters
    ----------
    filename : str 
        Full file name (with path) of the data file
    
    sensor_info : str | dict
        If str, it is the name of the json file containing experiment details. If dict, contains the details of the sensor parameters as a dictionary of values (see documentation for details on needed parameters).

    Returns
    -------
    df : pandas DataFrame  
        Pandas dataframe of the data in the framework friendly base csv format 
    """
    
    # global col_headers
    
    # data = read_idf(filename, 'START', missing=-1.0)
    # df = pd.DataFrame(columns=col_headers)

    # i = 0
    # for d in data:
    #     temp_dict = dict.fromkeys(col_headers)

    #     temp_dict['Timestamp'] = d['trackertime']
    #     temp_dict['StimulusName'] = ['stimulus_' + str(i)] * len(temp_dict['Timestamp'])
    #     temp_dict['EventSource'] = ['ET'] * len(temp_dict['Timestamp'])
    #     temp_dict['GazeLeftx'] = d['x']
    #     temp_dict['GazeRightx'] = d['x']
    #     temp_dict['GazeLefty'] = d['y']
    #     temp_dict['GazeRighty'] = d['y']
    #     temp_dict['PupilLeft'] = d['size']
    #     temp_dict['PupilRight'] = d['size']
    #     temp_dict['FixationSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
    #     temp_dict['SaccadeSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
    #     temp_dict['Blink'] = np.ones(len(temp_dict['Timestamp'])) * -1
        
    #     fix_cnt = 0
    #     sac_cnt = 0
    #     prev_end = 0
    #     for e in d['events']['Efix']:
    #         ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
    #         ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
    #         temp_dict['FixationSeq'][ind_start : ind_end + 1] = fix_cnt
    #         fix_cnt += 1
    #         if prev_end != ind_start:
    #             temp_dict['SaccadeSeq'][prev_end + 1 : ind_start + 1] = sac_cnt
    #             sac_cnt += 1
    #         prev_end = ind_end

    #     cnt = 0
    #     for e in d['events']['Eblk']:
    #         ind_start = np.where(temp_dict['Timestamp'] == e[0])[0][0]
    #         ind_end = np.where(temp_dict['Timestamp'] == e[1])[0][0]
    #         temp_dict['Blink'][ind_start : ind_end + 1] = cnt
    #         cnt += 1

    #     df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
    #     del(temp_dict)

    #     i += 1

    # df.to_csv(filename.split('.')[0] + '.csv') 
    # return df
    return


def csvToBaseET(filename, sensor_info):
    """
    """
    return


def csvToBaseEEG(filename, sensor_info, ch_names, stimulus_channel):
    """
    """

    if type(sensor_info) is str:
        with open(sensor_info) as f: 
            json_data = json.load(f)
        montage = json_data["Analysis_Params"]["EEG"]["Montage"]
        sfreq = sensor_info["Analysis_Params"]["EEG"]["Sampling_Freq"]
        eog = sensor_info["Analysis_Params"]["EEG"]["EOG"]

    elif type(sensor_info) is dict:
        montage = sensor_info["EEG"]["Montage"]
        sfreq = sensor_info["EEG"]["Sampling_Freq"]
        eog = sensor_info["EEG"]["EOG"]

    ch_types = {key:"eeg" for key in ch_names}
    for i in eog:
        ch_types[i] = "eog"
    ch_types[stimulus_channel] = "stim"
    
    df = pd.read_csv(filename, usecols=ch_names)

    info = mne.create_info(ch_names=ch_names, ch_types = [v for v in ch_types.values()], sfreq=sfreq, montage=montage)
    raw = mne.io.RawArray(data=df, info=info, verbose=False)

    return


def edfToBase(filename, sensor_info):
    """
    Bridge function that converts European Data Format (EDF and EDF+) raw EEG data files to the base CSV format that the framework uses.

    Parameters
    ----------
    filename : str 
        Full file name (with path) of the data file
    
    sensor_info : str | dict
        If str, it is the name of the json file containing experiment details. If dict, contains the details of the sensor parameters as a dictionary of values (see documentation for details on needed parameters).

    Returns
    -------
    raw : mne Raw array   
        Raw array of mne containing the EEG data (Not preloaded. Data will actually be loaded into memory only while analysisng each stimulus.)
     """

    if type(sensor_info) is str:
        with open(sensor_info) as f: 
            json_data = json.load(f)
        montage = json_data["Analysis_Params"]["EEG"]["Montage"]
        sfreq = sensor_info["Analysis_Params"]["EEG"]["Sampling_Freq"]
        eog = sensor_info["Analysis_Params"]["EEG"]["EOG"]

    elif type(sensor_info) is dict:
        montage = sensor_info["EEG"]["Montage"]
        sfreq = sensor_info["EEG"]["Sampling_Freq"]
        eog = sensor_info["EEG"]["EOG"]

    raw = mne.io.read_raw_edf(filename, montage=montage, eog=eog, verbose=False)
    
    data = raw.get_data(picks=[-1])
    
    stim_data = np.array(data[-1], dtype=int)

    diff = stim_data[1:] - stim_data[:-1]
    start = np.where(diff > 0)[0]
    end = np.where(diff < 0)[0]

    if start[0] > end[0]:
        start = np.hstack((0, start))
    
    if end[-1] < start[-1]:
        start = np.hstack((end, len(stim_data)))

    joining_char = "/"
    components = filename.split(joining_char)
    f_name = components[-1].split(".")[0]
    stim_indices_file = "question_indices/" + f_name + "_eeg.pickle"

    pickle.dump((start, end), open(stim_indices_file, "wb"))

    return raw


def brainvisionToBase(filename, sensor_info):
    """
    """
    return

def neuroscanToBase(filename, sensor_info):
    """
    """
    return


def egiToBase(filename, sensor_info):
    """
    """
    return


def mffToBase(filename, sensor_info):
    """
    """
    return


def eeglabToBase(filename, sensor_info):
    """
    """
    return


def matlabToBase(filename, sensor_info):
    """
    """
    return


def eximiaToBase(filename, sensor_info):
    """
    """
    return


def convertToBase(filename, sensor_type, sensor_info=None):
    file_type = filename.split('.')[1]

    if sensor_type == 'EyeTracker':
        if file_type == 'asc':
            return eyeLinkToBase(filename, sensor_info)
        
        elif file_type == 'idf':
            return smiToBase(filename, sensor_info)

        elif file_type == 'csv':
            return csvToBaseET(filename, sensor_info)
        
        else:
            print("Sorry " + sensor_type + " data format not supported!\n")
            return
    
    elif sensor_type == 'EEG':
        if file_type == 'csv':
            return csvToBaseEEG(filename, sensor_info)
        
        if file_type in ['edf', 'gdf', 'bdf']:
            return edfToBase(filename, sensor_info)
        
        elif file_type == 'vhdr':
            return brainvisionToBase(filename, sensor_info)
        
        elif file_type == 'cnt':
            return neuroscanToBase(filename, sensor_info)
        
        elif file_type == 'egi':
            return egiToBase(filename, sensor_info)
        
        elif file_type == 'mff':
            return mffToBase(filename, sensor_info)
        
        elif file_type == 'nxe':
            return eximiaToBase(filename, sensor_info)
        
        elif file_type == 'mat':
            return matlabToBase(filename, sensor_info)

        else:
            print("Sorry " + sensor_type + " data format not supported!\n")
            return
    
    else:
        print("Sorry " + sensor_type + " not supported!\n")


data = convertToBase("/home/upamanyu/Documents/NTU_Creton/Paul/Data/x2012_01_006_facehousepriming.bdf", "EEG", sensor_info={"EEG" : {"Sampling_Freq":1000, "Montage":'biosemi128', "EOG":None}})

print(type(data))