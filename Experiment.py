#Experiment class
import numpy as np
from datetime import datetime
from scipy import stats
from Sensor import Sensor
from Subject import Subject


import json

class Experiment:

	stimuli_classes = ["Relevant", "GenLie", "General", "Alpha"]

	def __init__(self, name,json_file,sensors):
		self.name = name #string
		self.json_file = json_file #string
		self.sensors = sensors
		self.columns = self.columnsArrayInitialisation()
		self.stimuli = self.stimuliArrayInitialisation() #dict of names of stimuli demarcated by category
		self.subjects = self.subjectArrayInitialisation() #list of subject objects
		self.meta_matrix_dict = (np.array(len(self.subjects), dtype=object), 
								{"sacc_count" : np.array((len(self.subjects), 4), dtype=object), 
								"sacc_dur" : np.array((len(self.subjects), 4), dtype=object),
								"blink_count" : np.array((len(self.subjects), 4), dtype=object),
								"ms_count" : np.array((len(self.subjects), 4), dtype=object), 
								"ms_duration" : np.array((len(self.subjects), 4), dtype=object), 
								"pupil_size" : np.array((len(self.subjects), 4), dtype=object),
								"fixation_count" : np.array((len(self.subjects), 4), dtype=object)})


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
		
		for i, sub in enumerate(self.subjects):
			sub.subjectAnalysis()
			meta_matrix_dict[0][i] = sub.subj_type
			for ind, j in enumerate(stimuli_classes):
				for k in Sensor.meta_cols[0]:
					meta_matrix_dict[1][k][i, ind] = sub.metadata_aggregate[j][k]

	
	def compareStimuliClasses(self, stim_class_1, stim_class_2, metadata):
		subj_types = self.meta_matrix_dict[0]
		for meta in metadata:
			stim_1_data = self.meta_matrix_dict[1][meta][:, stimuli_classes.index(stim_class_1)]
			stim_2_data = self.meta_matrix_dict[1][meta][:, stimuli_classes.index(stim_class_2)]

			innocent_data = stim_1_data[np.where(subj_types == "Innocent")[0]]
			guilty_data = stim_1_data[np.where(subj_types == "Guilty")[0]]

			(f_score_innocent, p_value_innocent) = stats.f_oneway([id for id in innocent_data])
			(f_score_guilty, p_value_guilty) = stats.f_oneway([id for id in guilty_data])




print("Start")
a = datetime.now()
exp = Experiment("Exp1", "trial_data.json", ["Eye Tracker"])
b = datetime.now()
print("End")
print((b-a).seconds)