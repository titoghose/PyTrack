#Subject class

import json
import pandas as pd
import numpy as np
from Stimulus import Stimulus
from Sensor import Sensor
from sqlalchemy import create_engine
import os
import pickle
from datetime import datetime
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt


class Subject:


	def __init__(self, name, subj_type, stimuli_names, columns, json_file, sensors, database):
		print(name)
		a = datetime.now()
		self.stimuli_names = stimuli_names
		self.name = name
		self.subj_type = subj_type
		self.stimulus = self.stimulusDictInitialisation(stimuli_names,columns,json_file,sensors, database) #dictionary of objects of class stimulus demarcated by categories
		self.control_data = self.getControlData(columns, json_file, sensors, database)
		self.aggregate_meta = {}
		b = datetime.now()
		print("Total time for subject: ", (b-a).seconds, "\n")


	def dataExtraction(self, columns,json_file, database):
		'''
		Extracts the required columns from the data base and returns a pandas datastructure

		Input:
		1.	name_of_database: [string] name of the database
		2.	columns: [list] list of the names of the columns of interest

		Output:
		1.	df: [pandas datastructure] contains the data of columns of our interest
		'''
		string = 'SELECT '

		index = 0

		a = datetime.now()

		for name in columns:

			if index == 0:
				string = string + name
				index = index + 1

			else:   
				string = string + ',' + name
				index = index + 1

		string = string + ' FROM "' + self.name + '"'
		df = pd.read_sql_query(string, database)

		b = datetime.now()
		print("Query: ", (b-a).seconds)
		
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

	
	def stimulusDictInitialisation(self, stimuli_names,columns,json_file,sensors, database):

		'''
		Creates  a list of objects of class Stimuli

		Input:
		1.	stimuli_names:[list] list of names of different stimulus
		2.	columns: [list] list of names of the columns of interest

		Output:
		1.	stimulus_object_dict: [dictionary] dictionary of objects of class stimulus ordered by category
		'''	

		if os.path.isfile('question_indices/' + self.name + '.pickle') == True:
			flag = 1

			pickle_in = open('question_indices/' + self.name + '.pickle',"rb")
			question_indices_dict = pickle.load(pickle_in)

		else:
			flag = 0

			question_indices_dict = {}
			stimulus_column = self.dataExtraction(["StimulusName"],json_file, database)


		data = self.dataExtraction(columns,json_file, database)

		stimulus_object_dict = {}

		for category in stimuli_names:
			
			stimulus_object_list = []

			for stimulus_name in stimuli_names[category]: 
				if flag == 1:
					[start_time, end_time, roi_time] = question_indices_dict[stimulus_name]  
				else:
					start_time, end_time, roi_time = self.timeIndexInitialisation("StimulusName",stimulus_name, stimulus_column)

					question_indices_dict[stimulus_name] = [start_time, end_time, roi_time]	

				stimuli_data = data[start_time : end_time+1]

				# print(self.name)
				# print(stimulus_name)
				# print(stimuli_data)

				stimulus_object = Stimulus(stimulus_name, category, sensors, stimuli_data, start_time, end_time, roi_time, json_file)

				stimulus_object_list.append(stimulus_object)

			stimulus_object_dict[category] = stimulus_object_list
		
		if flag == 0:	
			pickle_out = open('question_indices/' + self.name + '.pickle',"wb")
			pickle.dump(question_indices_dict, pickle_out)
			pickle_out.close()
	
		return stimulus_object_dict


	def getControlData(self, columns, json_file, sensors, database):
		'''

		This function returns the average value of control data (alpha questions) for the purpose of standardisation

		'''
		
		if os.path.isfile('control_values/' + self.name + '.pickle') == True:
			pickle_in = open('control_values/' + self.name + '.pickle',"rb")
			control = pickle.load(pickle_in)

		else:
			with open(json_file) as json_f:
				json_data = json.load(json_f)

			control_questions = {"Control" : json_data["Control_Questions"]}

			control_q_objects = self.stimulusDictInitialisation(control_questions, columns, json_file, sensors, database)

			control = {"sacc_count" : 0, 
						"sacc_duration" : 0,
						"blink_count" : 0,
						"ms_count" : 0,
						"ms_duration" : 0,
						"fixation_count" : 0,
						"ms_vel" : 0,
						"ms_amplitude" : 0,
						"peak_pupil" : 0,
						"time_to_peak_pupil" : 0}

			temp = []

			cnt = 0
			for cqo in control_q_objects["Control"]:
				if cqo.data != None:
					cnt += 1
					cqo.findEyeMetaData()
					for c in control:
						control[c] += np.mean(cqo.sensors[Sensor.sensor_names.index("EyeTracker")].metadata[c])

			for c in control:
				control[c] /= cnt

			control.update({"response_time" : 0})
			pickle_out = open('control_values/' + self.name + '.pickle',"wb")
			pickle.dump(control, pickle_out)
			pickle_out.close()

		return control


	def subjectVisualize(self):
		fig = plt.figure()
		
		def stimFunction(text):
			stim_t = text.split(",")[0].strip(" ")
			stim_n = text.split(",")[1].strip(" ")
			stim = self.stimulus[stim_t][self.stimuli_names[stim_t].index(stim_n)]
			stim.visualize()

		tax1 = plt.axes([0.25, 0.25, 0.50, 0.50])
		tb1 = TextBox(tax1, "Stimulus [type,name]", initial="alpha,Alpha1")
		
		try:
			tb1.on_submit(stimFunction)
		except:
			print("ERROR: STIMULUS NOT FOUND")

		plt.show()


	def subjectAnalysis(self,average_flag,standardise_flag):

		'''


		'''
		for st in self.stimulus:
			self.aggregate_meta.update({st : {}})
			for mc in Sensor.meta_cols[0]:
				self.aggregate_meta[st].update({mc : []})

		cnt = 0
		temp_pup_size = []
		for s in self.stimulus:
			for stim in self.stimulus[s]:
				if stim.data != None:
					stim.findEyeMetaData()
					
					# Normalizing by subtracting control data
					for cd in self.control_data:
						if(standardise_flag):
							self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], (stim.sensors[Sensor.sensor_names.index("EyeTracker")].metadata[cd] - self.control_data[cd])))
						else:
							self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], stim.sensors[Sensor.sensor_names.index("EyeTracker")].metadata[cd]))

					temp_pup_size.append(stim.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["pupil_size"])

			max_len = max([len(x) for x in temp_pup_size])
			
			temp_agg_pup_size = np.ma.empty((max_len, len(temp_pup_size)))
			temp_agg_pup_size.mask = True
			
			for ind, tps in enumerate(temp_pup_size):
				temp_agg_pup_size[:len(tps), ind] = tps

			temp_agg_pup_size = temp_agg_pup_size.mean(axis=1)
			self.aggregate_meta[s]["pupil_size"] = temp_agg_pup_size.data

			temp_pup_size = []

		if(average_flag):	
			for s in self.stimulus:
				for cd in self.control_data:
					self.aggregate_meta[s][cd] = np.array([np.mean(self.aggregate_meta[s][cd], axis=0)])
