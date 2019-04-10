from Experiment import Experiment
from datetime import datetime

print("Start")
a = datetime.now()

exp = Experiment("Exp1", "trial_data.json", ["EyeTracker", "EEG"], manual_eeg = True)

exp.analyse(standardise_flag = False, average_flag = False)

# exp.visualizeData()

b = datetime.now()
print("End")
print("Total time taken: ", (b-a).seconds)