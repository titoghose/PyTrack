import mne
import json
import pandas as pd
import numpy as np
from Stimulus import Stimulus, groupHeatMap
from Sensor import Sensor
from sqlalchemy import create_engine
import os
import pickle
import tkinter as tk
from datetime import datetime
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
from functools import partial


class SubjectVisualize:
	
	def __init__(self, master, subj_name, stimuli, json_file=None, viz_type="individual", sub_list=None):
		self.json_file = json_file
		self.sub_list = sub_list
		self.viz_type = viz_type

		self.root = master
		self.v = tk.IntVar()
		self.v.set(3)

		self.subject_window = tk.Toplevel(master)
		self.subject_window.title(subj_name)

		self.subject_window.resizable(False, False)

		self.stimuli = stimuli

		plot_type_frame = tk.Frame(self.subject_window)
		
		tk.Radiobutton(plot_type_frame, text="Gaze Heat Map", padx=20, variable=self.v, value=3).pack(side="left", anchor=tk.W)
		
		if viz_type == "individual":
			tk.Radiobutton(plot_type_frame, text="Live Plot", padx=20, variable=self.v, value=1).pack(side="right", anchor=tk.W)
			tk.Radiobutton(plot_type_frame, text="Fixation Plot", padx=20, variable=self.v, value=2).pack(side="right", anchor=tk.W)
		
		plot_type_frame.pack(side="top")

		stim_names_frame = tk.Frame(self.subject_window)

		for stim_type in self.stimuli:
			
			live_plot_frame = tk.Frame(stim_names_frame, width=30, height=30)
			live_plot_frame.grid_propagate(False)

			text = tk.Text(live_plot_frame, width=20)
			text.tag_configure("bold", font="Helvetica 12 bold")
			text.insert("end", stim_type, "bold")
			text.insert("end", "\n\n")
			
			for i, stim in enumerate(self.stimuli[stim_type]):
				func = partial(self.button_click, stim, i)
				bt = tk.Button(live_plot_frame, text=stim.name, width=10, command=func)
				text.window_create(tk.END, window=bt)
				text.insert(tk.END, "\n")
			
			scroll = tk.Scrollbar(live_plot_frame, orient="vertical")
			scroll.config(command=text.yview)

			text.configure(yscrollcommand=scroll.set)
			
			text.pack(side="left", expand=True, fill="both")
			scroll.pack(side="right", fill="y")
			live_plot_frame.pack(side="right", expand=True)	
		
		stim_names_frame.pack(side="bottom")

	def button_click(self, stim, stim_num=-1):
		if self.v.get() == 1:
			stim.visualize()

		elif self.v.get() == 2:
			stim.gazePlot()
		
		elif self.v.get() == 3:
			if self.viz_type == "group":
				stim_name = {stim.stim_type : stim_num}
				# self.subject_window.destroy()
				groupHeatMap(self.sub_list, stim_name, self.json_file)

			else:
				stim.gazeHeatMap()


class Subject:

	def __init__(self, name, subj_type, stimuli_names, columns, json_file, sensors, database):
		print(name)
		a = datetime.now()
		self.sensors = sensors
		self.stimuli_names = stimuli_names
		self.name = name
		self.subj_type = subj_type
		self.json_file = json_file
		self.stimulus = self.stimulusDictInitialisation(stimuli_names, columns, json_file, sensors, database) 
		self.control_data = self.getControlData(columns, json_file, sensors, database)
		self.aggregate_meta = {}
		b = datetime.now()
		print("Total time for subject: ", (b-a).seconds, "\n")


	def dataExtraction(self, columns, json_file, database):
		"""
		Extracts the required columns from the data base and returns a pandas datastructure

		Input:
		1.	name_of_database: [string] name of the database
		2.	columns: [list] list of the names of the columns of interest

		Output:
		1.	df: [pandas datastructure] contains the data of columns of our interest
		"""
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

		"""
		This function that will retireve the index of the start, end and roi of a question

		Input:
		1.	stimulus_column_name: [string] Name of the column where the stimuli names are present 
		2.	stimulus_name: [string] Name of the stimulus 

		Output:
		1.	start:[integer] the index of the start of a queation
		2.	end:[integer] the index of the end of a question
		3.	roi:[integer] the index when the eye lands on the region of interest
		"""

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

		return start, end, roi

	
	def stimulusDictInitialisation(self, stimuli_names, columns, json_file, sensors, database):

		"""
		Creates  a list of objects of class Stimuli

		Input:
		1.	stimuli_names:[list] list of names of different stimulus
		2.	columns: [list] list of names of the columns of interest

		Output:
		1.	stimulus_object_dict: [dictionary] dictionary of objects of class stimulus ordered by category
		"""	

		if not os.path.isdir('./question_indices/'):
			os.makedirs('./question_indices/')

		if os.path.isfile('question_indices/' + self.name + '.pickle') == True:
			flag = 1

			pickle_in = open('question_indices/' + self.name + '.pickle',"rb")
			question_indices_dict = pickle.load(pickle_in)

		else:
			flag = 0

			question_indices_dict = {}
			stimulus_column = self.dataExtraction(["StimulusName"],json_file, database)


		data = self.dataExtraction(columns, json_file, database)

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

				stimulus_object = Stimulus(stimulus_name, category, sensors, stimuli_data, start_time, end_time, roi_time, json_file, self.name)

				stimulus_object_list.append(stimulus_object)

			stimulus_object_dict[category] = stimulus_object_list
		
		del data

		if flag == 0:	
			pickle_out = open('question_indices/' + self.name + '.pickle',"wb")
			pickle.dump(question_indices_dict, pickle_out)
			pickle_out.close()
	
		return stimulus_object_dict


	def getControlData(self, columns, json_file, sensors, database):
		"""
		This function returns the average value of control data (alpha questions) for the purpose of standardisation
		"""

		control = dict()
		for sen in self.sensors:
			control.update({sen:dict()})
			for meta in Sensor.meta_cols[sen]:
				control[sen].update({meta: 0})

		with open(json_file) as json_f:
			json_data = json.load(json_f)

		if "Control_Questions" in json_data:
			if not os.path.isdir('./control_values/'):
				os.makedirs('./control_values/')
			
			if os.path.isfile('control_values/' + self.name + '.pickle') == True:
				pickle_in = open('control_values/' + self.name + '.pickle',"rb")
				control = pickle.load(pickle_in)

			else:
				cnt = 0
				for stim_cat in self.stimuli_names:
					for stim_ind, stim in enumerate(self.stimuli_names[stim_cat]):
						if stim in json_data["Control_Questions"]:
							cqo = self.stimulus[stim_cat][stim_ind]
							if cqo.data != None:
								cnt += 1
								cqo.findEyeMetaData()
								for sen in self.sensors:
									for c in control[sen]:
										control[sen][c] = np.hstack((control[sen][c], cqo.sensors[sen].metadata[c]))
				
				for sen in self.sensors:
					for c in control[sen]:
						control[sen][c] = np.mean(control[sen][c])

				control.update({"response_time" : 0})
				pickle_out = open('control_values/' + self.name + '.pickle',"wb")
				pickle.dump(control, pickle_out)
				pickle_out.close()
		
		return control


	def subjectVisualize(self, master, viz_type="individual", sub_list=None):
		"""
		"""
		sub_viz = SubjectVisualize(master, self.name, self.stimulus, json_file=self.json_file, viz_type=viz_type, sub_list=sub_list)


	def subjectAnalysis(self,average_flag,standardise_flag):
		"""
		"""
		
		for st in self.stimulus:
			self.aggregate_meta.update({st : {}})
			for sen in self.sensors:
				for mc in Sensor.meta_cols[sen]:
					self.aggregate_meta[st].update({mc : []})

		for s in self.stimulus:
			for stim in self.stimulus[s]:
				if stim.data != None:
					stim.findEyeMetaData()
					for sen in self.sensors:
						# Normalizing by subtracting control data
						for cd in Sensor.meta_cols[sen]:
							if(standardise_flag):
								self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], (stim.sensors["EyeTracker"].metadata[cd] - self.control_data[sen][cd])))
							else:
								self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], stim.sensors["EyeTracker"].metadata[cd]))

						# temp_pup_size.append(stim.sensors["EyeTracker"].metadata["pupil_size"])

		if(average_flag):	
			for s in self.stimulus:
				for sen in self.sensors:
					for cd in Sensor.meta_cols[sen]:
						self.aggregate_meta[s][cd] = np.array([np.mean(self.aggregate_meta[s][cd], axis=0)])