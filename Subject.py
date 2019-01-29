#Subject class

import json
import pandas as pd
from Stimulus import Stimulus
from sqlalchemy import create_engine
import os
import pickle

class Subject:


	def __init__(self, name,subj_type,stimuli_names,columns,json_file,sensors):

		self.name = name
		self.subj_type = subj_type
		self.stimulus = self.stimulusDictInitialisation(stimuli_names,columns,json_file,sensors) #dictionary of objects of class stimulus demarcated by categories
		self.control_data = self.getControlData(columns, json_file, sensors)

	def dataExtraction(self, columns,json_file):

		'''
		Extracts the required columns from the data base and returns a pandas datastructure

		Input:
		1.	name_of_database: [string] name of the database
		2.	columns: [list] list of the names of the columns of interest

		Output:
		1.	df: [pandas datastructure] contains the data of columns of our interest
		'''

		with open(json_file) as json_f:
			json_data = json.load(json_f)

		name_of_database = json_data["Database_name"]

		extended_name = "sqlite:///" + name_of_database
		database = create_engine(extended_name)

		string = 'SELECT '

		index = 0

		for name in columns:

			if index == 0:
				string = string + name
				index = index + 1

			else:   
				string = string + ',' + name
				index = index + 1

		string = string + ' FROM "' + self.name + '"'

		df = pd.read_sql_query(string, database)

		return df
		

	def timeIndexInitialisation(self, stimulus_column_name,stimulus_name, df):

		'''
		This function that will retireve the index of the start, end and roi of a question

		Input:
		1.	stimulus_column_name: [string] Name of the column where the stimuli names are present 
		2.	stimulus_name: [string] Name of the stimulus 

		Output:
		1.	start:[integer] the index of the start of a queation
		2.	end:[integer] the index of the end of a question
		3.	roi:[integer] the index when the eye lands on the region of interest
		'''

		index = df[df[stimulus_column_name] == stimulus_name].index

		try:
			start = min(index)
			end = max(index)
			if end - start < 1000:
				start = -1
				end = -1
			roi = -1
		except:
			start = -1
			end = -1
			roi = -1

		return start,end,roi

	
	def stimulusDictInitialisation(self, stimuli_names,columns,json_file,sensors):

		'''
		Creates  a list of objects of class Stimuli

		Input:
		1.	stimuli_names:[list] list of names of different stimulus
		2.	columns: [list] list of names of the columns of interest

		Output:
		1.	stimulus_object_dict: [dictionary] dictionary of objects of class stimulus ordered by category
		'''	

		if os.path.isfile('question_indices.pickle') == True:

			flag = 1

			pickle_in = open("question_indices.pickle","rb")
			question_indices_dict = pickle.load(pickle_in)

		else:
			flag = 0

			question_indices_dict = {}
			stimulus_column = self.dataExtraction(["StimulusName"],json_file)


		data = self.dataExtraction(columns,json_file)

		stimulus_object_dict = {}

		for category in stimuli_names:
			
			stimulus_object_list = []

			for stimulus_name in stimuli_names[category]: 
				
				if flag == 1:
					[start_time,end_time,roi_time] = question_indices_dict[stimulus_name]  
				else:
					start_time,end_time,roi_time = self.timeIndexInitialisation("StimulusName",stimulus_name, stimulus_column)

					question_indices_dict[stimulus_name] = [start_time, end_time, roi_time]	

				stimuli_data = data[start_time : end_time+1]

				# print(self.name)
				# print(stimulus_name)
				# print(stimuli_data)

				stimulus_object = Stimulus(stimulus_name, category, sensors, stimuli_data, start_time, end_time, roi_time)

				stimulus_object_list.append(stimulus_object)

			stimulus_object_dict[category] = stimulus_object_list
		
		if flag == 0:	
			pickle_out = open("question_indices.pickle","wb")
			pickle.dump(question_indices_dict, pickle_out)
			pickle_out.close()
	
		return stimulus_object_dict


	def getControlData(self, columns, json_file, sensors):

		with open(json_file) as json_f:
			json_data = json.load(json_f)

		control_questions = {"Alpha" : json_data["Control_Questions"]}

		control_q_objects = self.stimulusDictInitialisation(control_questions, columns, json_file, sensors)

		control = {"sacc_count" : 0, 
					"sacc_duration" : [],
					"blink_count" : 0,
					"ms_count" : 0,
					"ms_duration" : [],
					"pupil_size" : [],
					"fixation_count" : 0}

		for cqo in control_q_objects:
			cqo.findEyeMetaData()

