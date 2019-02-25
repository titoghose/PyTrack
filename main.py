from Experiment import Experiment
from datetime import datetime

print("Start")
a = datetime.now()
exp = Experiment("Exp1", "trial_data.json", ["Eye Tracker"])
# exp.analyse(average_flag = False,standardise_flag = False)
# exp.visualizeData()
print("\t\t\t\tResults after averaging and standardising")
b = datetime.now()
print("End")
print("Total time taken: ", (b-a).seconds)