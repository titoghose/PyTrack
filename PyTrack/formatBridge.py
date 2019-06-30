# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from PyTrack.etDataReader import read_edf, read_idf, read_tobii

def getColHeaders():
    """Function to return the column headers for the *PyTrack* base format data representation.

    """
    return ["Timestamp", "StimulusName", "EventSource", "GazeLeftx", "GazeRightx", "GazeLefty", "GazeRighty", "PupilLeft", "PupilRight", "FixationSeq", "SaccadeSeq", "Blink", "GazeAOI"]


def toBase(et_type, filename, stim_list=None, start='START', stop=None, eye='B'):
    """Bridge function that converts SMI, Eyelink or Tobii raw eye tracking data files to the base CSV format that the framework uses.

    Parameters
    ----------
    et_type : str {"smi","tobii", "eyelink"}
        Which eye tracker

    filename : str
        Full file name (with path) of the data file

    stim_list : list (str)
        Name of stimuli as a list of strings. If there are n trials/events found in the data, the length of stim_list should be n containing the names of stimuli for each trial/event.

    start : str
        Marker for start of event in the .asc file. Default value is 'START'.

    stop : str
        Marker for end of event in the .asc file. Default value is None. If None, new event/trial will start when start trigger is detected again.

    Returns
    -------
    df : pandas DataFrame
        Pandas dataframe of the data in the framework friendly base csv format
    """

    col_headers = getColHeaders()

    print("Converting file to Pandas CSV: ", filename.split("/")[-1])

    if et_type == "smi":
        data = read_idf(filename, start=start, stop=stop, missing=-1.0)
    elif et_type == "eyelink":
        data = read_edf(filename, start=start, stop=stop, missing=-1.0, eye=eye)
    elif et_type == "tobii":
        data = read_tobii(filename, start=start, stop=stop, missing=-1.0)

    df = pd.DataFrame(columns=col_headers)

    i = 0
    for d in data:
        temp_dict = dict.fromkeys(col_headers)

        temp_dict['Timestamp'] = d['trackertime']

        if stim_list == None:
            temp_dict['StimulusName'] = ['stimulus_' + str(i)] * len(temp_dict['Timestamp'])
        else:
            temp_dict['StimulusName'] = [stim_list[i]] * len(temp_dict['Timestamp'])

        temp_dict['EventSource'] = ['ET'] * len(temp_dict['Timestamp'])
        temp_dict['GazeLeftx'] = d['x_l']
        temp_dict['GazeRightx'] = d['x_r']
        temp_dict['GazeLefty'] = d['y_l']
        temp_dict['GazeRighty'] = d['y_r']
        temp_dict['PupilLeft'] = d['size_l']
        temp_dict['PupilRight'] = d['size_r']
        temp_dict['FixationSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['SaccadeSeq'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['Blink'] = np.ones(len(temp_dict['Timestamp'])) * -1
        temp_dict['GazeAOI'] = np.ones(len(temp_dict['Timestamp'])) * -1


        if et_type != "eyelink":
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

        else:
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

        for m in d['events']['msg']:
            if 'Stim Key' in m[1]:
                temp_dict['StimulusName'] = [m[1].strip("\n").split(":")[1].strip(" ")] * len(temp_dict['Timestamp'])
                break

        df = df.append(pd.DataFrame.from_dict(temp_dict, orient='index').transpose(), ignore_index=True, sort=False)
        del(temp_dict)

        i += 1
        print("Progress: %.2f" % ((i*100)/len(data)), end="\r")

    print()
    return df


def convertToBase(filename, sensor_type, device, stim_list=None, start='START', stop=None, eye='B'):
    """Master function that calls the different converter functions to convert to bas data format.

    Internally invoked by `generateCompatibleFormats <#formatBridge.generateCompatibleFormat>`_.

    Parameters
    ----------
    filename : str
        Full path of the data file.
    sensor_type : str {'EyeTracker'}
        Type of sensor. Supports only 'EyeTracker' for now.
    device : str {'eyelink', 'smi', 'tobii'}
        Make of the sensor. Must be a type of eye tracker.
    stim_list : list (str) | None
        If ``None`` (default) the name of stimuli for a data file wil be generated sequentially as ["stim1", "stim2", "stim2", ...]. If it is a list of str, then the stimulus names will be taken from this list in the given order. Hence, the length of stim_list and number of events/trials/simuli markers in the data should be the same.
    start : str
        The start of event marker in the data (Defaults to 'START').
    stop : str
        The end of event marker in the data (Defaults to ``None``). If ``None``, start of new event will be considered as end of previous event.
    eye : str {'B','L','R'}
		Which eye is being tracked? Deafults to 'B'-Both. ['L'-Left, 'R'-Right, 'B'-Both]

    Returns
    -------
    pandas DataFrame
        The extracted data in the base format.

    """

    if sensor_type == 'EyeTracker':
        try:
            return toBase(device, filename, stim_list, start=start, stop=stop, eye=eye)
        except Exception as e:
            print("Sorry " + sensor_type + " data format not supported! The following exception was thrown: \n")
            print(e)
            return

    else:
        print("Sorry " + sensor_type + " not supported!\n")
        return


def db_create(data_path, source_folder, database_name, dtype_dictionary=None, na_strings=None):
    """Create a SQL database from a csv file

	Parameters
	---------
	data_path: string
		Path to the directory where the database is to be located
	source_folder: string
		Name of folder that contains the csv files
	database_name: string
		Name of the SQL database that is to be created
	dtype_dictionary: dictionary
		Dictionary mapping the names of the columns ot their data type
	na_strings: list
		Is a list of the strings that are to be considered as null value

	"""

    all_files = os.listdir(source_folder)

    newlist = []
    for names in all_files:
        if names.endswith(".csv"):
            newlist.append(names)

    database_extension = "sqlite:///" + data_path +  database_name + ".db"
    database = create_engine(database_extension)

    for file in newlist:

        print("Creating sql file for: ", file)

        file_name = source_folder + "/" + file

        file_name_no_extension = file.split(".")[0]
        table_name = file_name_no_extension.replace(' ','_') #To ensure table names dont haves spaces

        try:
            query = "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
            dummy = database.execute(query)
            conversion = pd.DataFrame(dummy.fetchall())
            conversion.columns = dummy.keys()

            for value in conversion.name:

                if value == table_name:
                    query = "DROP TABLE " + table_name
                    dummy = database.execute(query)
                    print("Dropped table as it existed previously and reinstantiated it: ", table_name)
                    break
        except ValueError:
            print("A new SQL database is being created")

        chunksize = 100000
        i = 0
        j = 1
        for df in pd.read_csv(file_name, chunksize=chunksize, iterator=True, na_values = na_strings, dtype=dtype_dictionary):

            #SQL columns ideally should not have ' ', '/', '(', ')'

            df = df.rename(columns={c: c.replace(' ', '_') for c in df.columns})
            df = df.rename(columns={c: c.replace('/', '') for c in df.columns})
            df = df.rename(columns={c: c.replace('(', '') for c in df.columns})
            df = df.rename(columns={c: c.replace(')', '') for c in df.columns})

            df.index += j
            i+=1
            df.to_sql(table_name, database, if_exists='append',index = True, index_label = "Index")
            j = df.index[-1] + 1


def generateCompatibleFormat(exp_path, device, stim_list_mode="NA", start='START', stop=None, eye='B', reading_method="SQL"):
    """Function to convert data into the base format before starting analysis and visualization.

    The function creates a directory called 'csv_files' inside the `Data` folder and stores the converted csv files in it. If `reading_method` is specified as 'SQL' then an SQL database is created inside the 'Data' folder but the user need not worry about it.

    Parameters
    ----------
    exp_path : str
        Absolute path to the experiment folder. If the path is a folder, the framework will assume its being run in the *Experiment Design* mode. If it is a path to a single data file, then the *Stand-alone Design* mode is assumed.
    device : str {'eyelink', 'smi', 'tobii'}
        Make of the sensor.
    stim_list_mode : str {'NA', 'common', 'diff'}
        See the `Using PyTrack <https://pytrack-ntu.readthedocs.io/en/latest/Introduction.html#using-pytrack>`_ in the `Introduction <https://pytrack-ntu.readthedocs.io/en/latest/Introduction.html#>`_ for details on which of the three to suply.
    start : str
        The start of event marker in the data (Defaults to 'START').
    stop : str
        The end of event marker in the data (Defaults to ``None``). If ``None``, start of new event will be considered as end of previous event.
    eye : str {'B','L','R'}
		Which eye is being tracked? Deafults to 'B'-Both. ['L'-Left, 'R'-Right, 'B'-Both]
    reading_method : str {'CSV', 'SQL'}
        'SQL' (default) reading method is faster but will need extra space. This affects the internal functioning of he framework and the user can leave it as is.

    """

    exp_path = exp_path.replace("\\", "/")

    if os.path.isdir(exp_path):

        exp_info = exp_path + "/" + exp_path.split("/")[-1] + ".json"
        data_path = exp_path + "/Data/"
        if not os.path.isdir(data_path + "/csv_files/"):
            os.makedirs(data_path + "/csv_files/")

        stim = None

        if stim_list_mode == "common":
            stim = np.loadtxt(data_path + "/" + "stim_file.txt", dtype=str)


        for f in os.listdir(data_path):
            if os.path.isdir(data_path + "/" + f):
                continue

            if f.split(".")[-1] not in ["csv", "asc", "txt", "tsv"]:
                continue

            print("Converting to base csv format    : ", f)

            if stim_list_mode == "diff":
                stim = np.loadtxt(data_path + "/stim/" + f.split(".")[0] + ".txt", dtype=str)

            df = convertToBase(data_path + "/" + f, sensor_type='EyeTracker', device=device, stim_list=stim, start=start, stop=stop, eye=eye)
            df.to_csv(data_path + "/csv_files/" + f.split(".")[0] + ".csv")

        source_folder = data_path + "/csv_files/"

        with open(exp_info, "r") as json_f:
            json_data = json.load(json_f)

        if reading_method == "SQL":
            db_create(data_path, source_folder, json_data["Experiment_name"])

    else:
        data_path = exp_path
        print("Converting to base csv format: ", data_path.split("/")[-1])

        stim = None
        if stim_list_mode != "NA":
            stim = np.loadtxt("stim_file.txt", dtype=str)

        df = convertToBase(data_path, sensor_type='EyeTracker', device=device, stim_list=stim, start=start, stop=stop, eye=eye)
        df.to_csv(data_path.split(".")[0] + ".csv")