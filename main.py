from Experiment import Experiment
from datetime import datetime

print("Start")
a = datetime.now()
exp = Experiment("Exp1", "trial_data_copy.json", ["EyeTracker", "EEG"])

exp.analyse(standardise_flag = False)

#exp.visualizeData()

b = datetime.now()
print("End")
print("Total time taken: ", (b-a).seconds)