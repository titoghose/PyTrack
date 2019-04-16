from Stimulus import Stimulus
import pandas as pd
import numpy as np

df = pd.read_csv("/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 12.csv", usecols=["GazeLeftx", "GazeLefty", "GazeRightx", "GazeRighty", "PupilLeft", "PupilRight", "FixationSeq", "StimulusName", "EventSource"])

stim_name = "Wallet-1-5"

df = df.replace(to_replace=r'Unnamed:*', value=float(-1), regex=True)
df = df.iloc[np.where(df["StimulusName"] == stim_name)[0]]

sensor_dict = {"EyeTracker":{"Sampling_Freq":1000}}

stim = Stimulus(data=df, sensor_names=sensor_dict, name=stim_name)

# stim.findMicrosaccades(plot_ms=True)
stim.gazePlot()
stim.gazeHeatMap()
