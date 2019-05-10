
#I need to become better. I need to start working harder. Atleast 20 minutes without distractions, OK? Now going for lunch.

'''
#Calculating duration pass

no_q_roi = 0
no_r_roi = 0
pre_roi = -1

previous_flag = -1
present_flag = -1

first_pass = -1
second_pass = -1


length = len(df["GazeAOI"])

for i in range(length):

	s1 = df["GazeAOI"][i]

	#Calculation of present flag

	if(pd.isnull(s1)):
		present_flag = -1
	elif('Q' in s1):
		present_flag = 1
	else:
		present_flag = -1

	if(present_flag != previous_flag and present_flag == 1):

		if(df["GazeAOI"][i:i+4].nunique() == 1):

			no_q_roi += 1
			duration_pass = i 


	if(present_flag != previous_flag and previous_flag == 1):
			print("Duration: ", i - duration_pass)

	if(df["GazeAOI"][i:i+4].nunique() == 1):
		previous_flag = present_flag

print(datetime.now())

print("Question stimuli: ", no_q_roi)
'''

'''
def analyse(self, standardise_flag=False, average_flag=True, stat_test=True, parameter_list = set(), between_factor_list = ["Subject_type"], within_factor_list = ["Stimuli_type"], statistical_test = "Mixed_anova"):
'''

'''
This function carries out the required statistical analysis technique for the specified indicators/parameters

Input:
standardise_flag: [Boolean]
average_flag: [Boolean] 
stat_test: [Boolean] 
parameter_list: [list of strings] List of the different indicators/parameters (Pupil_size, Blink_rate) on which statistical analysis is to be performed 
between_factor_list: [list of strings] List of between group factors
within_factor_list: [list of strings] List of within group factors
statistical_test: [string] Name of the statistical test that has to be performed


Output: 

'''

'''	
	#Definig the meta_matrix_dict data structure
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


	if stat_test:

		#NTBC: For each column parameter (WHAT DOES THIS MEAN)
		p_value_table = pd.DataFrame()
		flag = 1

		#To find the number of subjects in Group 1 (Innocent) and Group 2 (Guilty)
		with open(self.json_file, "r") as json_f:
			json_data = json.load(json_f)
			num_inn = len(json_data["Subjects"]["innocent"])
			num_guil = len(json_data["Subjects"]["guilty"])
		
		num_samples = 2 #Number of participants randomly picked from each group
		num_runs = 1 #Number of times the experiment anlaysis is run (Results may differ in each run as the participants chosen will be different)


		for sen in self.sensors: #For each type of sensor
			for i in range(num_runs): #For each run

				#sub_indices is a list of indices, where each index refers to a subject
				sub_indices = np.hstack((random.sample(range(0, num_inn), num_samples), random.sample(range(num_inn, num_inn + num_guil), num_samples)))
				#print(sub_indices)

				# sub_indices = np.hstack((random.sample(range(0, 2), 2), random.sample(range(2, 4), 2)))
				# print(sub_indices)

				head_row = ["Indices"]
				csv_row = [str(sub_indices)]

				with open(json_file) as json_f:
					json_data = json.load(json_f)

				for meta in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
					if meta == "pupil_size" or meta == "pupil_size_downsample" or meta not in parameter_list:
						continue

					print("\n\n")
					print("\t\t\t\tAnalysis for ",meta)	

					#For the purpose of statistical analysis, a pandas dataframe needs to be created that can be fed into the statistical functions
					#The columns required are - meta (indicator), the between factors (eg: Subject type or Gender), the within group factor (eg: Stimuli Type), Subject name/id

					#Defining the list of columns required for the statistical analysis
					column_list = [meta]

					column_list.append(between_factor_list)
					column_list.append(within_factor_list)
					column_list.append("subject")

					data =  pd.DataFrame(columns=column_list)


					#For each subject
					for sub_index in sub_indices:
						
						sub = self.subjects[sub_index] #sub refers to the object of the respective subject whose data is being extracted

						#For each Question Type (NTBC: FIND OUT WHAT THE AGGREGATE_META CONTAINS)
						for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):

							#NTBC: Ignore the picture stimuli for the time being
							if stimuli_type not in ['alpha', 'general', 'general_lie', 'relevant']:
								continue

							#Value is an array (NTBC : Is it always an array or can it also be a single value?)	
							value_array = self.meta_matrix_dict[1][meta][sub_index,stimuli_index]

							try:					
								for value in value_array:

									row = []

									row.append(value)

									#add the between group factors (need to be defined in the json file)
									for param in between_factor_list:
										row.append(json_data["Subjects"][sub.subj_type][sub.name][param])

									#NTBD: Please change sub.name to required value
									#DO NOT HAVE ACCESS TO THE NAME OF THE STIMULI, SO NEED TO FIGURE OUT HOW THE WITHIN GROUP 
									#PARAMETERS NEED TO BE ACCESSED 
									for param in within_factor_list:
										row.append(json_data["Stimuli"][stimuli_type][sub.name][param])

									row.append(stimuli_type)
									row.append(sub.name)

									#NTBC: Checking condition if value is nan for error checking
									if(np.isnan(value)):
										print(row)

									#Instantiate into the pandas dataframe
									data.loc[len(data)] = row

							except:
								print("Value array for ", stimuli_type, " is empty")
						

						#Data shaping - end

					#Depending on the parameter, choose the statistical test to be done

					if(statistical_test == "Mixed_anova"):

						print(meta, "Mixed anova")
						mixed_anova_calculation(meta, data, subject_parameters, stimuli_parameters)

					elif(statistical_test == "RM_anova"):

						print(meta, "RM Anova")
						rm_anova(meta, data, subject_parameters, stimuli_parameters)

					elif(statistical_test == "ttest"):

						print(meta, "t test")
						ttest_calculation(meta, data, subject_parameters, stimuli_parameters)

					#4. NTBD : Genlie vs Relevant comparison to be done

							
#Mixed anova
def mixed_anova_calculation(meta, data, subject_parameters, stimuli_parameters):

	#Mixed Anova

	column_values = []

	try:
		aov = pg.mixed_anova(dv=meta, within=stimuli_paramters, between=subject_parameters, subject = 'subject', data=data)
		pg.print_table(aov)

		posthocs = pg.pairwise_ttests(dv=meta, within=stimuli_paramters, between=subject_parameters, subject='subject', data=data)
		pg.print_table(posthocs)

	except:
		print("problem with anova\n\n")
		print(data)

#Repeated measures anova
def rm_anova(meta, data):

	aov = pg.rm_anova(dv=meta, within='stimuli_type', subject = 'subject', data=innocent_data)
	pg.print_table(aov)


#Comparison of relevant to general lie using rm anova and t test (specific to CRETON data)
def relevant_genlie_comparison():


	innocent_data = data.loc[(data['individual_type'] == 'innocent') & ((data['stimuli_type'] == 'relevant') | (data['stimuli_type'] == 'general_lie'))]

	aov = pg.rm_anova(dv=meta, within='stimuli_type', subject = 'subject', data=innocent_data)
	posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', subject='subject', data=innocent_data)

	guilty_data = data.loc[(data['individual_type'] == 'guilty') & ((data['stimuli_type'] == 'relevant') | (data['stimuli_type'] == 'general_lie'))]

	aov = pg.rm_anova(dv=meta, within='stimuli_type', subject = 'subject', data=guilty_data)
	posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', subject='subject', data=guilty_data)

	column_values.append(posthocs['p-unc'][0])

#T test 
def ttest_calculation(meta, data):

	posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', between='individual_type', subject='subject', data=data)
	pg.print_table(aov)
'''	

'''
import numpy as np

list_values = ["all"]

print(list_values.count("all"))
print(list_values.count("arvind"))

y = np.random.randint(0, 2, size=30)
print(y)

'''



#Testing time access tradeoffk

'''
# Testing if index column can be accessed

import pandas as pd
import os
from sqlalchemy import create_engine
from datetime import datetime
import json
import numpy as np
import pickle


json_file = "/home/arvind/Desktop/ntuExperiment/trial_data.json"

with open(json_file) as json_f:
	json_data = json.load(json_f)


stimuli_data = json_data["Stimuli"]
subject_data = json_data["Subjects"]

#name = "Innocent_Subject_06"

a = datetime.now()

data_dict = {}

for k in stimuli_data:

	for stimulus_name in stimuli_data[k]:

		database_name = "Lie_trial"

		database = "sqlite:///" + database_name + ".db"
		csv_database = create_engine(database)

		
		try:
			query = "SELECT Index_column, StimulusName FROM Guilty_Subject_06 WHERE StimulusName = '" + stimulus_name + "'"
			
			dummy = csv_database.execute(query)
			conversion = pd.DataFrame(dummy.fetchall())
			conversion.columns = dummy.keys()
		except:
			continue	

b = datetime.now()
print("Where condition in SQL", (b-a).seconds)


a = datetime.now()

for Subject_type in subject_data:
	for name in subject_data[Subject_type]:

		print("Name: ", name)

		c = datetime.now()

		database_name = "lie_detection_database"
		database = "sqlite:///" + database_name + ".db"
		csv_database = create_engine(database)

		#query = "SELECT 'Index', StimulusName FROM Guilty_Subject_06 WHERE StimulusName = '" + stimulus_name + "'"

		query = "SELECT StimulusName, PupilLeft, PupilRight FROM " + name

		dummy = csv_database.execute(query)
		conversion = pd.DataFrame(dummy.fetchall())
		conversion.columns = dummy.keys()

		for k in stimuli_data:

			for stimulus_name in stimuli_data[k]:

				try:
					filter1 = conversion[conversion["StimulusName"] == stimulus_name]
					print(filter1)
					break
				except:
					print(stimulus_name)

			break	

		d = datetime.now()

		print("Where condition in Pandas", (d-c))


		e = datetime.now()

		database_name = "lie_detection_database"
		database = "sqlite:///" + database_name + ".db"
		csv_database = create_engine(database)

		query = "SELECT StimulusName FROM " + name 

		dummy = csv_database.execute(query)
		conversion = pd.DataFrame(dummy.fetchall())
		conversion.columns = dummy.keys()


		if os.path.isfile('question_indices/' + name + '.pickle') == True:

			pickle_in = open('question_indices/' + name + '.pickle',"rb")
			question_indices_dict = pickle.load(pickle_in)


		for k in stimuli_data:
			for stimulus_name in stimuli_data[k]:

				[start_time, end_time, roi_time] = question_indices_dict[stimulus_name]
				try:
					filter1 = conversion[start_time:end_time+1]
				except:
					print(stimulus_name)	

		f = datetime.now()

		print("Pandas extracting by row number", (f-e))

		g = datetime.now()

		database_name = "lie_detection_database"
		database = "sqlite:///" + database_name + ".db"
		csv_database = create_engine(database)

		query = "SELECT StimulusName FROM " + name 

		dummy = csv_database.execute(query)
		conversion = pd.DataFrame(dummy.fetchall())
		conversion.columns = dummy.keys()

		if os.path.isfile('question_indices/' + name + '.pickle') == True:

			pickle_in = open('question_indices/' + name + '.pickle',"rb")
			question_indices_dict = pickle.load(pickle_in)


		#print(len(conversion))
		#print(conversion.columns)		

		for k in stimuli_data:
			for stimulus_name in stimuli_data[k]:

				[start_time, end_time, roi_time] = question_indices_dict[stimulus_name]
				filter1 = conversion[conversion["Index_column"] > start_time]

				#print(filter1)

				filter2 = filter1[filter1["Index_column"] < end_time]

				#print(filter2)

				break

			break

		h = datetime.now()

		print("Pandas extracting using Index column", (h-g))

		database_name = "lie_detection_database"
		database = "sqlite:///" + database_name + ".db"
		csv_database = create_engine(database)

		i = datetime.now()

		query = "SELECT StimulusName, PupilLeft FROM " + name 

		dummy = csv_database.execute(query)
		conversion = pd.DataFrame(dummy.fetchall())
		conversion.columns = dummy.keys()

		j = datetime.now()

		print("Selecting the entire column", j-i)

		flag = -1

		for k in stimuli_data:

				for stimulus in stimuli_data[k]:

					#print(stimulus)

					if(flag == -1):
						flag = 1
						string_stimulus = "'" + stimulus + "'"

					else:
						string_stimulus = string_stimulus + ", " + "'" + stimulus + "'"


		k = datetime.now()

		query = "SELECT StimulusName, PupilLeft FROM " + name + " WHERE StimulusName IN ("  + string_stimulus + ")"
		# ORDER BY Index_Column

		dummy = csv_database.execute(query)
		conversion = pd.DataFrame(dummy.fetchall())
		conversion.columns = dummy.keys()

		l = datetime.now()

		print("Selecting specific rows", l-k)

		print("\n\n")

b = datetime.now()

print("Total time taken is: ", b-a)

'''

import pickle
import pingouin as pg
import pandas
import statsmodels.api as sm


#Testing of the logistic regression

def logistic_regression_trial(independent_parameter_list):


	pickle_in = open('logistic_regression.pickle',"rb")
	data = pickle.load(pickle_in)

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

	logit = sm.Logit(dependent_column, X, fit_intercept = True)


	result = logit.fit()

	print(result.summary2())					



#logistic_regression_trial(independent_parameter_list = ["response_time", "pupil_size", "time_to_peak_pupil", "peak_pupil", "pupil_mean", "alpha", "general", "general_lie", "relevant"])

logistic_regression_trial(independent_parameter_list = ["response_time", "pupil_size", "time_to_peak_pupil", "peak_pupil", "pupil_mean"])