import sys
import os
sys.path.append(os.path.abspath("../PyTrack"))
sys.path.append(os.path.abspath("../PyTrack/PyTrack"))
print(sys.path)

import json
from formatBridge import generateCompatibleFormat
from Experiment import Experiment

# function to convert data to generate database in base format for experiment done using EyeLink on both eyes and the stimulus name specified in the message section
generateCompatibleFormat(exp_path=os.path.abspath("NTU_Experiment/Data"),
                        device="eyelink", 
                        stim_list_mode='NA', 
                        start='start_trial', 
                        stop='stop_trial', 
                        eye='B')
                    
# Creating an object of the Experiment class
exp = Experiment(json_file=os.path.abspath("NTU_Experiment/NTU_Experiment.json"))

# Instantiate the meta_matrix_dict of an Experiment to find and extract all features from the raw data
exp.metaMatrixInitialisation(standardise_flag=False, 
                              average_flag=False)

# Calling the function for the statistical analysis of the data
# file_creation=True. Hence, the output of the data used to run the tests and the output of the tests will be stored in in the 'Results' folder inside your experiment folder
exp.analyse(parameter_list={"all"}, 
            between_factor_list=["Subject_type"], 
            within_factor_list=["Stimuli_type"], 
            statistical_test="Mixed_anova", 
            file_creation=True)


# Calling the function for advanced statistical analysis of the data 
# file_creation=True. Hence, the output of the data used to run the tests and the output of the tests will be stored in in the 'Results' folder inside your experiment folder

#############################################################
## 1. Running anova on advanced between and within factors ##
#############################################################
exp.analyse(parameter_list={"all"}, 
            between_factor_list=["Subject_type", "Gender"],
            within_factor_list=["Stimuli_type", "Brightness"],
            statistical_test="anova", 
            file_creation=True)

#############################################################
## 2. Running no tests. Just storing analysis data in Results folder ##
#############################################################
exp.analyse(statistical_test="None", 
            file_creation=True)

# In case you want the data for a particular participant/subject as a dictionary of values, use this

subject_name = "sub_222" #specify your own subject's name (must be in json file)
stimulus_name = "Alpha1" #specify your own stimulus name (must be in json file)

# Access metadata dictionary for particular subject and stimulus
single_meta = exp.getMetaData(sub=subject_name, 
                              stim=stimulus_name)

# Access metadata dictionary for particular subject and averaged for stimulus types
agg_type_meta = exp.getMetaData(sub=subject_name, 
                                 stim=None)