import mne
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


	def __init__(self, name, subj_type, stimuli_names, columns, json_file, sensors, database, manual_eeg):
		print(name)
		a = datetime.now()
		self.sensors = sensors
		self.stimuli_names = stimuli_names
		self.name = name
		self.subj_type = subj_type
		self.manual_eeg = manual_eeg
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
		df = df.replace(to_replace=r'Unnamed:*', value=float(-1), regex=True)

		b = datetime.now()
		print("Query: ", (b-a).seconds)
		
		return df
		

	def timeIndexInitialisation(self, stimulus_column_name, stimulus_name, df):

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

		if "EEG" in self.sensors and self.manual_eeg:
			data = self.manualEEGArtefactRemovalSubject(data, json_file)

		stimulus_object_dict = {}

		for category in stimuli_names:
			columns
			stimulus_object_list = []

			for stimulus_name in stimuli_names[category]: 
				if flag == 1:
					[start_time, end_time, roi_time] = question_indices_dict[stimulus_name]  
				else:
					start_time, end_time, roi_time = self.timeIndexInitialisation("StimulusName",stimulus_name, stimulus_column)

					question_indices_dict[stimulus_name] = [start_time, end_time, roi_time]	

				stimuli_data = data[start_time : end_time+1]

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
		control = dict()
		for sen in self.sensors:
			control.update({sen:dict()})
			for meta in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
				control[sen].update({meta: 0})

		with open(json_file) as json_f:
			json_data = json.load(json_f)

		if "Control_Questions" in json_data:
			if os.path.isfile('control_values/' + self.name + '.pickle') == True:
				pickle_in = open('control_values/' + self.name + '.pickle',"rb")
				control = pickle.load(pickle_in)

			else:
				control_questions = {"Control" : json_data["Control_Questions"]}
				control_q_objects = self.stimulusDictInitialisation(control_questions, columns, json_file, sensors, database)

				cnt = 0
				for cqo in control_q_objects["Control"]:
					if cqo.data != None:
						cnt += 1
						cqo.findEyeMetaData()
						for sen in self.sensors:
							for c in control[sen]:
								control[sen][c] = np.hstack((control[sen][c], cqo.sensors[Sensor.sensor_names.index(sen)].metadata[c]))
				
				for sen in self.sensors:
					for c in control[sen]:
						control[sen][c] = np.mean(control[sen][c])

				control.update({"response_time" : 0})
				pickle_out = open('control_values/' + self.name + '.pickle',"wb")
				pickle.dump(control, pickle_out)
				pickle_out.close()
		
		return control


	def subjectVisualize(self):
		
		plt.figure()

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
		print(self.control_data)
		
		for st in self.stimulus:
			self.aggregate_meta.update({st : {}})
			for sen in self.sensors:
				for mc in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
					self.aggregate_meta[st].update({mc : []})

		for s in self.stimulus:
			for stim in self.stimulus[s]:
				if stim.data != None:
					stim.findEyeMetaData()
					for sen in self.sensors:
						# Normalizing by subtracting control data
						for cd in self.control_data[sen]:
							if(standardise_flag):
								self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], (stim.sensors[Sensor.sensor_names.index("EyeTracker")].metadata[cd] - self.control_data[sen][cd])))
							else:
								self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], stim.sensors[Sensor.sensor_names.index("EyeTracker")].metadata[cd]))

						# temp_pup_size.append(stim.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["pupil_size"])

		# print(str(self.aggregate_meta))

		if(average_flag):	
			for s in self.stimulus:
				for sen in self.sensors:
					for cd in self.control_data[sen]:
						self.aggregate_meta[s][cd] = np.array([np.mean(self.aggregate_meta[s][cd], axis=0)])

	def manualEEGArtefactRemovalSubject(self, data, json_file):

		if os.path.isfile(".icaRejectionLog.p"):
			with open(".icaRejectionLog.p", "rb") as f:
				ica_rejection_dict = pickle.load(f)
		else:
			ica_rejection_dict = dict()

		eeg_rows = np.where(data.EventSource.str.contains("Raw EEG Epoc"))[0]

		with open(json_file) as f:
			json_data = json.load(f)
		
		# Getting eeg preprocessing parameters from experiment json file
		low = json_data["Analysis_Params"]["EEG"]["Low_Freq"]
		high = json_data["Analysis_Params"]["EEG"]["High_Freq"]
		sampling_freq = json_data["Analysis_Params"]["EEG"]["Sampling_Freq"]
		montage = json_data["Analysis_Params"]["EEG"]["Montage"]
		
		eeg_cols = [*json_data["Columns_of_interest"]["EEG"]]
		ch_names = []
		for channel in eeg_cols:
			ch_names.append([i for i in Sensor.eeg_montage[montage] if i.upper() in channel.upper()][0])

		del(json_data)

		eeg_df = data[eeg_cols]
		eeg_data = np.transpose(eeg_df.values)
		info = mne.create_info(ch_types=["eeg"]*len(ch_names), ch_names=ch_names, sfreq=sampling_freq, verbose=False, montage=montage)
		raw = mne.io.RawArray(data=eeg_data[:, eeg_rows], info=info, verbose=False)

		# Apply bandpass filter to eeg data
		raw = raw.filter(l_freq=low, h_freq=high)

		ica = mne.preprocessing.ICA(method='fastica', random_state=1, verbose=False)
		ica.fit(raw, verbose=False)
		
		if self.name in ica_rejection_dict:
			flag = input("Use previously rejected components? (Y/N)")
			if flag == 'Y' or flag == 'y':
				ica.exclude = ica_rejection_dict[self.name]
			else:
				ica.plot_components(inst=raw)
		else:
			ica.plot_components(inst=raw)
		
		ica_rejection_dict.update({self.name: ica.exclude})
		with open(".icaRejectionLog.p", "wb") as f:
			pickle.dump(ica_rejection_dict, f)

		raw_temp = raw.copy()
		ica.apply(raw)

		raw_temp.plot(n_channels=len(eeg_cols), duration=8, scalings='auto', show=True, title="Before ICA")
		raw.plot(n_channels=len(eeg_cols), duration=8, scalings='auto', show=True, title="After ICA")
		plt.show()

		eeg_data[:, eeg_rows] = raw.get_data()
		data[eeg_cols] = np.transpose(eeg_data)

		return data
		