from formatBridge import generateCompatibleFormat
from Experiment import Experiment
from datetime import datetime

print("Start")
a = datetime.now()

# generateCompatibleFormat("test_formats.json", data_path="/home/upamanyu/Documents/NTU_Creton/Paul/Data/EyeTracker")

exp = Experiment("test_formats.json")

# exp.analyse(standardise_flag = False, average_flag = False, stat_test=False)
# print(exp.subjects[0].aggregate_meta)

exp.visualizeData()

b = datetime.now()
print("End")
print("Total time taken: ", (b-a).seconds)