import sys
import os
sys.path.append(os.path.abspath("../PyTrack"))
sys.path.append(os.path.abspath("../PyTrack/PyTrack"))

from formatBridge import generateCompatibleFormat
from Experiment import Experiment
from Stimulus import Stimulus
import pandas as pd
import numpy as np
import json
import unittest

class TestMethods(unittest.TestCase):

    def testExperimentDesign(self):
        check = 0
        try:
            generateCompatibleFormat(exp_path=os.path.abspath("tests/NTU_Experiment"),
                                device="eyelink",
                                stim_list_mode='NA',
                                start='start_trial',
                                stop='stop_trial',
                                eye='B')
            check = 1
        finally:
            self.assertEqual(check, 1)

        check = 0
        try:
            exp = Experiment(json_file=os.path.abspath("tests/NTU_Experiment/NTU_Experiment.json"))
            check = 1
        finally:
            self.assertEqual(check, 1)

        check = 0
        try:
            exp.metaMatrixInitialisation(standardise_flag=False,
                                    average_flag=False)
            check = 1
        finally:
            self.assertEqual(check, 1)

        check = 0
        try:
            exp.analyse(parameter_list={"all"},
                        between_factor_list=["Subject_type", "Gender"],
                        within_factor_list=["Stimuli_type"],
                        statistical_test="anova",
                        file_creation=True)

            exp.analyse(parameter_list={"all"},
                        statistical_test="anova",
                        file_creation=True)

            exp.analyse(parameter_list={"all"},
                        statistical_test="ttest",
                        file_creation=True)

            exp.analyse(parameter_list={"all"},
                        statistical_test="RM_anova",
                        file_creation=True)

            exp.analyse(statistical_test="None",
                        file_creation=True)

            check = 1
        finally:
            self.assertEqual(check, 1)

        check = 0
        try:
            subject_name = "sub_222"
            stimulus_name = "Alpha1"

            single_meta = exp.getMetaData(sub=subject_name, stim=stimulus_name)

            agg_type_meta = exp.getMetaData(sub=subject_name, stim=None)
            check = 1
        finally:
            self.assertEqual(check, 1)

    def testStandaloneDesign(self):
        check = 0
        try:
            generateCompatibleFormat(exp_path=os.path.abspath("tests/NTU_Experiment/Data/sub_222.asc"),
                            device="eyelink",
                            stim_list_mode='NA',
                            start='start_trial',
                            stop='stop_trial',
                            eye='B')

            generateCompatibleFormat(exp_path=os.path.abspath("tests/NTU_Experiment/smi_eyetracker_freeviewing.txt"),
                            device="smi",
                            stim_list_mode='NA',
                            start='12',
                            stop='99')

            temp_df = pd.read_csv(os.path.abspath("tests/NTU_Experiment/smi_eyetracker_freeviewing.csv"))
            del(temp_df)
            df = pd.read_csv(os.path.abspath("tests/NTU_Experiment/Data/sub_222.csv"))
            check = 1
        finally:
            self.assertEqual(check, 1)

        check = 0
        try:
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

            stim = Stimulus(path=os.path.abspath("tests/NTU_Experiment"),
                        data=df,
                        sensor_names=sensor_dict,
                        start_time=0,
                        end_time=6000)

            check = 1
        finally:
            self.assertEqual(check, 1)

        check = 0
        try:
            stim.findEyeMetaData()
            features = stim.sensors["EyeTracker"].metadata
            stim.findMicrosaccades(plot_ms=True)
            stim.gazePlot(show_fig=False, save_fig=True)
            stim.gazeHeatMap(show_fig=False, save_fig=True)
            stim.visualize(show=False)
            check = 1
        finally:
            self.assertEqual(check, 1)

if __name__ == '__main__':
    unittest.main()