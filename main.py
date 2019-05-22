from PyTrack.Experiment import Experiment
from datetime import datetime
import warnings


from PyTrack.formatBridge import generateCompatibleFormat

warnings.filterwarnings("ignore")

print("Start")

a = datetime.now()

'''
exp = Experiment("/home/arvind/Desktop/Experiment1/Experiment1.json","SQL")
exp.metaMatrixInitialisation()

#Testing analyse
exp.analyse()
exp.analyse(within_factor_list=["Stimuli_type", "C"], statistical_test="RM_anova")
exp.analyse(between_factor_list=["Subject_type", "G"], statistical_test = "anova")
exp.analyse(statistical_test = "ttest")
exp.analyse(between_factor_list = ["Subject_type", "G"],  statistical_test = "ttest")
exp.analyse(statistical_test = "ttest")
'''

'''
# function to convert data to generate database in base format for experiment done using EyeLink on both eyes and the stimulus name specified in the message section
generateCompatibleFormat(exp_path="/home/arvind/Desktop/Pytrack_testing/PyTrack_sample_data/NTU_Experiment",
                        device="eyelink",
                        stim_list_mode='NA',
                        start='start_trial',
                        stop='stop_trial',
                        eye='B')
'''
exp = Experiment("/home/arvind/Desktop/Experiment1/Experiment1.json")
exp.visualizeData()
exp.metaMatrixInitialisation(average_flag=True)

# #Testing analyse
# exp.analyse(statistical_test="Mixed_anova")
# exp.analyse(statistical_test = "anova")
# exp.analyse(statistical_test = "RM_anova")
# exp.analyse(statistical_test = "ttest")
exp.analyse(statistical_test = "welch_ttest", ttest_type=2)
#exp.analyse(within_factor_list=["Stimuli_type", "C"], statistical_test="RM_anova")
# exp.analyse(between_factor_list=["Subject_type", "G"], statistical_test = "anova")
# exp.analyse(statistical_test = "ttest")
# exp.analyse(between_factor_list = ["Subject_type", "G"],  statistical_test = "ttest")
# exp.analyse(statistical_test = "ttest")
