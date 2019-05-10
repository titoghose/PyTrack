from Experiment import Experiment
from datetime import datetime
import warnings


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

exp = Experiment("/home/arvind/Desktop/NTU_Experiment/NTU_Experiment.json","SQL")
#exp.visualizeData()
exp.metaMatrixInitialisation(average_flag = True)

#Testing analyse
exp.analyse(statistical_test = "Mixed_anova")
# exp.analyse(statistical_test = "anova")
# exp.analyse(statistical_test = "RM_anova")
# exp.analyse(statistical_test = "ttest")
# exp.analyse(within_factor_list=["Stimuli_type", "C"], statistical_test="RM_anova")
# exp.analyse(between_factor_list=["Subject_type", "G"], statistical_test = "anova")
# exp.analyse(statistical_test = "ttest")
# exp.analyse(between_factor_list = ["Subject_type", "G"],  statistical_test = "ttest")
# exp.analyse(statistical_test = "ttest")
