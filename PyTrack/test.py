import json
from sqlalchemy import create_engine
import pandas as pd
import os
import csv
import numpy as np

'''
	LOGISTIC FUNCTION TO BE MADE BETTER (ITS ACCURACY NOT CHECKED YET)
def logistic_regression_analysis(self, average_flag=False, standardise_flag=False,  independent_parameter_list=["all"]):
		"""Run a logistic regression on the data from several parameters

		Parameters
		----------
		standardise_flag: bool {``False``, ``True``}
			Indicates whether the data being considered need to be standardised (by subtracting the control values/baseline value) 		
		average_flag: bool {``False``, ``True``} 
			Indicates whether the data being considered should averaged across all stimuli of the same type
		independent_parameter_list = list of strings {["all"]}
			Is a list of the independent variables in the logistic regression equation

		"""

		#Defining the meta_matrix_dict data structure
		for sensor_type in Sensor.meta_cols:
			for meta_col in sensor_type:
				self.meta_matrix_dict[1].update({meta_col : np.ndarray((len(self.subjects), len(self.stimuli)), dtype=object)})

		#Instantiation of the meta_matrix_dict database		
		for sub_index, sub in enumerate(self.subjects):
			sub.subjectAnalysis(average_flag, standardise_flag)

			self.meta_matrix_dict[0][sub_index] = sub.subj_type

			for stim_index, stimuli_type in enumerate(sub.aggregate_meta):
				for meta in sub.aggregate_meta[stimuli_type]:
					self.meta_matrix_dict[1][meta][sub_index, stim_index] = sub.aggregate_meta[stimuli_type][meta]

		#Define a pandas dataframe with the required column names
		#NTBD add stimuli_type as a value as well 
		#(see if pingouin takes nominal factors as well, if not search for a package that does take nominal values for independent factors)

		if independent_parameter_list.count("all") == 1:

			independent_parameter_list = []

			for sen in self.sensors:
				for meta in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
					independent_parameter_list.append(meta)


		dataframe_columns = ["Subject_type"]

		dataframe_columns.extend(independent_parameter_list)

		dataframe_columns.append("Stimuli_type")

		data =  pd.DataFrame(columns=dataframe_columns)


		#To find the number of subjects in Group 1 (Innocent) and Group 2 (Guilty)
		with open(self.json_file, "r") as json_f:
			json_data = json.load(json_f)
			num_inn = len(json_data["Subjects"]["innocent"])
			num_guil = len(json_data["Subjects"]["guilty"])
		
		num_samples = 5 #Number of participants randomly picked from each group
		num_runs = 1 #Number of times the experiment anlaysis is run (Results may differ in each run as the participants chosen will be different)

		sub_indices = np.hstack((random.sample(range(0, num_inn), num_samples), random.sample(range(num_inn, num_inn + num_guil), num_samples)))
		
		for sub_index in sub_indices:
							
			sub = self.subjects[sub_index] #sub refers to the object of the respective subject whose data is being extracted

			#For each Question Type
			for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):

				if stimuli_type not in ['alpha', 'general', 'general_lie', 'relevant']:
					continue

				row = []

				if(sub.subj_type == "guilty"):
					row.append(1)
				elif(sub.subj_type == "innocent"):
					row.append(0)
				else:
					print("Unrecognized Subject type: ", sub.subj_type)		

				#NTBD: how to handle if all the parameters are required?
				
				for meta in independent_parameter_list:

					#NTBD: Lets assumes that value_array is always a single value and not a list

					value_array = self.meta_matrix_dict[1][meta][sub_index,stimuli_index]
					
					row.append(mean(value_array))

				row.append(stimuli_index)

				data.loc[len(data)] = row	

		#Convert the Stimuli_type into dummy variables

		stimuli_dummy_data = pd.get_dummies(data["Stimuli_type"])

		stimuli_columns = ['alpha', 'general', 'general_lie', 'relevant']
		stimuli_dummy_data.columns = stimuli_columns

		data = data.join(stimuli_dummy_data)

		print(independent_parameter_list)

		independent_parameter_list.extend(stimuli_columns)

		#Instantiate X and y values of the linear regression

		print(independent_parameter_list)

		X = data[independent_parameter_list]
		y = data["Subject_type"]

		dependent_column = []

		for i in range(len(y)):

			if(y[i] == 0.0):
				 dependent_column.append(0)
			elif(y[i] == 1.0):
				dependent_column.append(1)

		print(dependent_column)

		#Logistic regression through pingoiun

		print("\t\t\t\tPingouin logistic regression")


		results = pg.logistic_regression(X, dependent_column)

		print(results.round(2))

		#Logistic regression through statsmodel

		print("\t\t\t\tStatmodels logistic regression")

		logit = sm.Logit(dependent_column, X, fit_intercept = False)

		result = logit.fit()

		print(result.summary2())


'''

'''

json_file = "/home/arvind/Desktop/NTU_Experiment/NTU_Experiment.json"
path = "/home/arvind/Desktop/NTU_Experiment"

with open(json_file) as json_f:
	json_data = json.load(json_f)


name_of_database = json_data["Experiment_name"]
name_of_database = "trial_SQL"
extended_name = "sqlite:///" + path + "/Data/" + name_of_database + ".db"
database = create_engine(extended_name)


query = "SELECT PupilLeft From sub_222"
dummy = database.execute(query)
conversion = pd.DataFrame(dummy.fetchall())
conversion.columns = dummy.keys()
print("SQL: ",  conversion.shape[0])


csv_file = "/home/arvind/Desktop/NTU_Experiment/Data/csv_files/sub_222.csv"
df = pd.read_csv(csv_file)
print("CSV: ", df.shape[0])
print(df.columns)
'''

'''
def db_create(data_path, source_folder, database_name, dtype_dictionary=None, na_strings=None):
	"""Create a SQL database from a csv file

	Parameter
	---------

	source_folder: string
		Name of folder that contains the csv files
	database_name: string
		Name of the SQL database that is to be created
	dtype_dictionary: dictionary
		Dictionary mapping the names of the columns ot their data type 
	na_strings: list 
		Is a list of the strings that re to be considered as null value
	"""
	
	all_files = os.listdir(source_folder) 
	
	newlist = []
	for names in all_files:
		if names.endswith(".csv"):
			newlist.append(names)

	database_extension =  "sqlite:///" + data_path + database_name + ".db"
	database = create_engine(database_extension)

	for file in newlist:

		print("Creating sql table for: ", file)

		file_name = source_folder + "/" + file

		length = len(file)
		file_name_no_extension = file[:length-4]
		table_name = file_name_no_extension.replace(' ','_') #To ensure table names dont haves spaces

		#Check if table exists, if it does then drop it and reinitialise it

		try:
			query = "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
			dummy = database.execute(query)
			conversion = pd.DataFrame(dummy.fetchall())
			conversion.columns = dummy.keys()

			for value in conversion.name:

				if value == table_name:
					query = "DROP TABLE " + table_name
					dummy = database.execute(query)
					print("Dropped table as it existed previously and reinstantiated it: ", table_name)
					break
		except ValueError:
			note = "The database did not exist previously"

		chunksize = 100000
		i = 0
		j = 1
		for df in pd.read_csv(file_name, chunksize=chunksize, iterator=True, na_values = na_strings, dtype=dtype_dictionary):
			
			#SQL columns ideally should not have ' ', '/', '(', ')'

			df = df.rename(columns={c: c.replace(' ', '_') for c in df.columns})
			df = df.rename(columns={c: c.replace('/', '') for c in df.columns})
			df = df.rename(columns={c: c.replace('(', '') for c in df.columns})
			df = df.rename(columns={c: c.replace(')', '') for c in df.columns})

			df.index += j
			i+=1
			df.to_sql(table_name, database, if_exists='append',index = True, index_label = "Index")
			j = df.index[-1] + 1


db_create("/home/arvind/Desktop/","/home/arvind/Desktop/NTU_Experiment/Data/csv_files","trial_SQL")
'''

#<class 'dict'>
#<class 'list'>

"""
  "Subjects":{
	  "group1":{
		 "Innocent_Subject_05":{"G":"M","Y":"H"},
		 "Innocent_Subject_06":{"G":"F","Y":"L"},
		 "Innocent_Subject_07":{"G":"M","Y":"L"},
		 "Innocent_Subject_08":{"G":"M","Y":"H"},
		 "Innocent_Subject_09":{"G":"M","Y":"L"},
		 "Innocent_Subject_10":{"G":"F","Y":"H"}
	  },
	  "group2":{
		 "Guilty_Subject_06":{"G":"F","Y":"H"},
		 "Guilty_Subject_07":{"G":"M","Y":"H"},
		 "Guilty_Subject_08":{"G":"M","Y":"H"},
		 "Guilty_Subject_09":{"G":"M","Y":"L"},
		 "Guilty_Subject_10":{"G":"F","Y":"L"},
		 "Guilty_Subject_11":{"G":"M","Y":"L"}
	  }
   },
   "Stimuli":{
	  "Type_1":{
		 "Alpha1": {"I": "N"},
		 "Alpha2": {"I": "Y"},
		 "Alpha3": {"I": "N"},
		 "Alpha4": {"I": "Y"}
	  },
	  "Type_2":{
		 "Gen1": {"I": "Y"},
		 "Gen2": {"I": "Y"},
		 "Gen3": {"I": "N"},
		 "Gen4": {"I": "N"},
		 "Gen5": {"I": "Y"}
	  },
	  "Type_3":{
		 "GenLie3": {"I": "N"},
		 "GenLie4": {"I": "Y"},
		 "GenLie6": {"I": "N"},
		 "GenLie7": {"I": "N"}
	  },
	  "Type_4":{
		 "Relevant7": {"I": "N"},
		 "Relevant3": {"I": "Y"},
		 "Relevant16":{"I": "Y"},
		 "Relevant15":{"I": "N"},
		 "Relevant10":{"I": "Y"}
	  }
   },
"""


file_name = "anova.csv"
path = "/home/arvind/Desktop/Experiment1"

directory_path = path + "/Results"

if not os.path.isdir(directory_path):
	os.mkdir(directory_path)
else:
	print("Exists")

file_path = directory_path + "/" + file_name 

row = ['4', ' Danny', ' New York']
row1 = ['45', ' Danny', ' New York']

data = np.array([['','Col1','Col2'],
                ['Row1',1,2],
                ['Row2',3,4]])
                
pandas_dataframe = pd.DataFrame(data=data[1:,1:],index=data[1:,0],columns=data[0,1:])
print(pandas_dataframe)

csvFile = open(file_path, 'w')
writer = csv.writer(csvFile)

pandas_dataframe.to_csv(csvFile)
writer.writerow("\n")
writer.writerow(["Mixed Anova"])
pandas_dataframe.to_csv(csvFile)
#
csvFile.close()