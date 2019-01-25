#Experiment class

import json

class Experiment:

	def __init__(self, name,json_file,sensors):
		self.name = name #string
		self.json_file = json_file #string
		self.columns = columnsArrayInitialisation(self.json_file)
		self.stimuli = self.stimuliArrayInitialisation(json_file) #dict of names of stimuli demarcated by category
		self.subjects = subjectArrayInitialisation(self.json_file,self.stimuli,self.columns) #list of subject objects
		self.sensors = sensors
		

	def stimuliNameInitialisation(self.json_file):

		'''
		This functions instialises the dictionary 'stimuli' with the list of names of the different stimuli by category

		Input:
		json_file : [string]Name of the json file which contains details of the experiment

		Output:
		data_dict : [dictionary]Dictionary containing the names of the different stimuli categorised by type
		'''

		with open(json_file) as json_f:
			json_data = json.load(json_f)

		stimuli_data = json_data["Stimuli"]

		data_dict = {}

		for k in stimuli_data:
			data_dict[k] = stimuli_data[k]

		return data_dict


	def subjectArrayInitialisation(self.json_file,self.stimuli):

		'''
		This function initialises an list of objects of class Subject

		Input:
		json_file : [string]Name of the json file which contains details of the experiment
		stimuli : [dictionary] Dictionary containing the names of the stimulus ordred by category 

		Output:
		subject_list : [list] list of objects of class Subject
		'''

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		subject_list = []	

		subject_data = json_data["Subjects"]

		for k in subject_data:

			for subject_name in subject_data[k]:

				subject_object = subject(subject_name,k,self.stimuli)

				subject_list.append(subject_object)

		return subject_list

	def columnsArrayInitialisation(self.json_file):

		'''

		The functions extracts the names of the columns that are to analysed

		Input:

		json_file: [string]Name of the json file which contains details of the experiment

		Output:

		columns_list: [list]list of names of columns of interest
		'''

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		column_list = []

		for name in json_data["Columns_of_interest"]:

			column_list.append(name)

		return column_list