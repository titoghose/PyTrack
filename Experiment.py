#Experiment class


import json

class Experiment:

	def __init__(self, name,json_file,sensors):
		self.name = name
		self.json_file = json_file
		self.subjects = None
		self.stimuli = self.stimuliArrayInitialisation(json_file)
		self.sensors = sensors


	def stimuliNameInitialisation(self.json_file):

		'''

		This functions instialises the dictionary 'stimuli' with the list of names of the different stimuli by category

		Input:
		json_file : [string]Name of the json file which contains details of the experiment

		Output:
		data_dict : [dictionary]Dictionary containig the names of the different stimuli categorised by type

		'''

		with open(json_file) as json_f:
			json_data = json.load(json_f)


		stimuli_data = json_data["Stimuli"]


		data_dict = {}

		for k in stimuli_data:
			data_dict[k] = stimuli_data[k]

		return data_dict


	def subjectArrayInitialisation(self.json_file):

		'''

		This function initialises an array of objects of class Subject

		Input:
		json_file : [string]Name of the json file which contains details of the experiment

		Output:
		subject_list : [array] list of objects of class Subject

		'''

		with open(json_file) as json_f:
			json_data = json.load(json_f)

		subject_list = []	

		subject_data = json_data["Subjects"]

		for k in subject_data:

			for subject_name in subject_data[k]:

				subject_object = subject(subject_name,k)

				subject_list.append(subject_object)


		return subject_list




