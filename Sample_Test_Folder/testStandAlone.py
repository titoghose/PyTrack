import sys
sys.path.append("..")
sys.path.append("../PyTrack")

from Stimulus import Stimulus
from formatBridge import generateCompatibleFormat
import pandas as pd
import numpy as np

# function to convert data to generate csv file for data file recorded using EyeLink on both eyes and the stimulus name specified in the message section
generateCompatibleFormat(exp_path="NTU_Experiment/Data/sub_222.asc",
                        device="eyelink", 
                        stim_list_mode='NA', 
                        start='start_trial', 
                        stop='stop_trial', 
                        eye='B')

df = pd.read_csv("NTU_Experiment/Data/sub_222.csv")

# Dictionary containing details of recording. Please change the values according to your experiment. If no AOI is desired, set aoi_left values to (0, 0) and aoi_right to the same as Display_width and Display_height
sensor_dict = {
                  "EyeTracker":
                  {
                     "Sampling_Freq": 1000,
                     "Display_width": 1280,
                     "Display_height": 1024,
                     "aoi_left_x": 0,
                     "aoi_left_y": 0,
                     "aoi_right_x": 1280,
                     "aoi_right_y": 1024
                  }
               }

# Creating Stimulus object. See the documentation for advanced parameters.
stim = Stimulus(path="NTU_Experiment",
               data=df, 
               sensor_names=sensor_dict,
               start_time=0,
               end_time=6000)

# Some functionality usage. See documentation of Stimulus class for advanced use.
stim.findEyeMetaData()
features = stim.sensors["EyeTracker"].metadata  # Getting dictioary of found metadata/features
stim.findMicrosaccades(plot_ms=True)