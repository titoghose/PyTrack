#Experiment class

from Sensor import Sensor
from Subject import Subject

import numpy as np
from datetime import datetime
from scipy import stats
import json
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp
import pandas as pd

import pingouin as pg


class Experiment:

	def __init__(self, name,json_file,sensors):
		self.name = name #string
		self.json_file = json_file #string
		self.sensors = sensors
		self.columns = self.columnsArrayInitialisation()
		self.stimuli = self.stimuliArrayInitialisation() #dict of names of stimuli demarcated by category
		self.subjects = self.subjectArrayInitialisation() #list of subject objects
		self.meta_matrix_dict = (np.ndarray(len(self.subjects), dtype=str), 
								{"sacc_count" : np.ndarray((len(self.subjects), 4), dtype=object), 
								"sacc_duration" : np.ndarray((len(self.subjects), 4), dtype=object),
								"blink_count" : np.ndarray((len(self.subjects), 4), dtype=object),
								"ms_count" : np.ndarray((len(self.subjects), 4), dtype=object), 
								"ms_duration" : np.ndarray((len(self.subjects), 4), dtype=object), 
								"pupil_size" : np.ndarray((len(self.subjects), 4), dtype=object),
								"fixation_count" : np.ndarray((len(self.subjects), 4), dtype=object),
								"response_time" : np.ndarray((len(self.subjects), 4), dtype=object)})


	def stimuliArrayInitialisation(self):

		'''
		This functions instialises the dictionary 'stimuli' with the list of names of the different stimuli by category

		Input:
		1.	json_file : [string]Name of the json file which contains details of the experiment

		Output:
		1.	data_dict : [dictionary]Dictionary containing the names of the different stimuli categorised by type
		'''

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		stimuli_data = json_data["Stimuli"]

		data_dict = {}

		for k in stimuli_data:
			data_dict[k] = stimuli_data[k]

		return data_dict


	def subjectArrayInitialisation(self):

		'''
		This function initialises an list of objects of class Subject

		Input:
		1.	json_file : [string]Name of the json file which contains details of the experiment
		2.	stimuli : [dictionary] Dictionary containing the names of the stimulus ordred by category 

		Output:
		1.	subject_list : [list] list of objects of class Subject
		'''

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		subject_list = []	

		subject_data = json_data["Subjects"]

		for k in subject_data:

			for subject_name in subject_data[k]:

				subject_object = Subject(subject_name, k, self.stimuli, self.columns, self.json_file, self.sensors)

				subject_list.append(subject_object)

		return subject_list


	def columnsArrayInitialisation(self):

		'''

		The functions extracts the names of the columns that are to analysed

		Input:
		1.	json_file: [string]Name of the json file which contains details of the experiment

		Output:
		1.	columns_list: [list]list of names of columns of interest
		'''

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		column_list = []

		for name in json_data["Columns_of_interest"]:

			column_list.append(name)

		return column_list


	def analyse(self):
		cnt = 0
		
		for sub_index, sub in enumerate(self.subjects):
			sub.subjectAnalysis()

			self.meta_matrix_dict[0][sub_index] = sub.subj_type
			
			

			for stim_index, stimuli_type in enumerate(sub.aggregate_meta):
				for meta in sub.aggregate_meta[stimuli_type]:
					self.meta_matrix_dict[1][meta][sub_index, stim_index] = sub.aggregate_meta[stimuli_type][meta]

		#Lets assume the meta_matrix_dict is instantiated for non_temporal data

		#1. Make a pandas dataframe containing the required columns

		#Creation of the dataframe skeleton


		#Instantiation of values into data dataframe

		
		#For each column parameter
		for sensor_type in Sensor.meta_cols:
			for meta in sensor_type:
				if meta == "pupil_size" or meta == "sacc_count" or meta == "sacc_duration":
					continue

				data =  pd.DataFrame(columns=[meta,"stimuli_type","individual_type","subject"])

				#For each subject
				for sub_index, sub in enumerate(self.subjects):

					#For each Question Type
					for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):

						#Value is an array	
						value_array = self.meta_matrix_dict[1][meta][sub_index,stimuli_index]

						try:					
							for value in value_array:

								row = []

								row.append(value)
								row.append(stimuli_type)
								row.append(sub.subj_type)
								row.append(sub.name)

								#Instantiate into the pandas dataframe

								data.loc[len(data)] = row
						except:
							print(stimuli_type)
					#2.Run the anlaysis

					# Fits the model with the interaction term
					# This will also automatically include the main effects for each factor
					
				  
				print(data)


				# Compute the two-way mixed-design ANOVA
				aov = pg.mixed_anova(dv=meta, within='stimuli_type', between='individual_type', subject = 'subject', data=data)
				# Pretty printing of ANOVA summary
				pg.print_table(aov)

				posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', between='individual_type', subject='subject', data=data)
				pg.print_table(posthocs)


				'''
				model_statement = meta + ' ~ C(stimuli_type)+C(individual_type)' 


				model = ols(model_statement, data).fit()

				#3. Print the Results
				#Results for testing significance of overall model
				print("\n\n\n\t\t\t\t********Analysis for sensor: ", meta,"********")
				print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")
				print("Overall model is valid if p value is less than 0.05")
				print("\n\n")	
				
				#Results of the summary for the model (used to check autocorelation, normality, homoscedasticity)
				print(model.summary())
				print("\n\n")

				#Seeing the anova statistics for the independent variables (stimulus_type and individual_type) and the interaction effect
				print(sm.stats.anova_lm(model, typ= 2))
				print("A parameter is significant if the corresponding p value is less than 0.05")

				'''
				

print("Start")
a = datetime.now()
exp = Experiment("Exp1", "trial_data.json", ["Eye Tracker"])
exp.analyse()
b = datetime.now()
print("End")
print((b-a).seconds)