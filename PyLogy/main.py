from Experiment import Experiment
from datetime import datetime

print("Start")
a = datetime.now()

exp = Experiment("Exp1", "trial_data.json")

# exp.analyse(standardise_flag = False, average_flag = False, stat_test=False)
# print(exp.subjects[0].aggregate_meta)

exp.visualizeData()

b = datetime.now()
print("End")
print("Total time taken: ", (b-a).seconds)