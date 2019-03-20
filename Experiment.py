#Experiment class

from Sensor import Sensor
from Subject import Subject
import numpy as np
from datetime import datetime
from scipy import stats
import json
import pandas as pd
from sqlalchemy import create_engine
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons


class Experiment:


	def __init__(self, name, json_file, sensors):
		self.name = name #string
		self.json_file = json_file #string
		self.sensors = sensors
		self.columns = self.columnsArrayInitialisation()
		self.stimuli = self.stimuliArrayInitialisation() #dict of names of stimuli demarcated by category
		self.subjects = self.subjectArrayInitialisation() #list of subject objects
		self.meta_matrix_dict = (np.ndarray(len(self.subjects), dtype=str), dict())

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

		name_of_database = json_data["Database_name"]
		extended_name = "sqlite:///" + name_of_database
		database = create_engine(extended_name)

		for k in subject_data:

			for subject_name in subject_data[k]:

				subject_object = Subject(subject_name, k, self.stimuli, self.columns, self.json_file, self.sensors, database)

				subject_list.append(subject_object)

		database.dispose()

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

		for col_class in json_data["Columns_of_interest"]:
			for name in json_data["Columns_of_interest"][col_class]:
				column_list.append(name)

		return column_list


	def visualizeData(self):
		subject_names = [s.name for s in self.subjects]
		rax = plt.axes([0, 0, 1.00, 1.00])
		radio = RadioButtons(rax, subject_names)
		
		def subjFunction(label):
			subj_dict = {s.name:s for s in self.subjects}
			subj_dict[label].subjectVisualize()

		radio.on_clicked(subjFunction)
		plt.show()


	def analyse(self, average_flag=True, standardise_flag=False, stat_test=True):
		
		for sensor_type in Sensor.meta_cols:
			for meta_col in sensor_type:
				self.meta_matrix_dict[1].update({meta_col : np.ndarray((len(self.subjects), len(self.stimuli)), dtype=object)})

		for sub_index, sub in enumerate(self.subjects):
			sub.subjectAnalysis(average_flag, standardise_flag)

			self.meta_matrix_dict[0][sub_index] = sub.subj_type

			for stim_index, stimuli_type in enumerate(sub.aggregate_meta):
				for meta in sub.aggregate_meta[stimuli_type]:
					self.meta_matrix_dict[1][meta][sub_index, stim_index] = sub.aggregate_meta[stimuli_type][meta]

		if stat_test:
			#For each column parameter
			for sensor_type in Sensor.meta_cols:
				for meta in sensor_type:
					if meta == "pupil_size" or meta == "sacc_count" or meta == "sacc_duration" or meta == "pupil_mean_list":
						continue

					print("\t\t\t\tAnalysis for ",meta)

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

					# Compute the two-way mixed-design ANOVA
					print(data[meta][:20])
					
					aov = pg.mixed_anova(dv=meta, within='stimuli_type', between='individual_type', subject = 'subject', data=data)
					# Pretty printing of ANOVA summary
					pg.print_table(aov)

					posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', between='individual_type', subject='subject', data=data)
					pg.print_table(posthocs)
