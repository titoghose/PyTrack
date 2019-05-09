# -*- coding: utf-8 -*-

import os
import json
import pickle
import tkinter as tk
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np

from PyTrack.Stimulus import Stimulus, groupHeatMap
from PyTrack.Sensor import Sensor

class SubjectVisualize:

	def __init__(self, master, subj_name, stimuli, json_file=None, viz_type="individual", sub_list=None):
		self.json_file = json_file
		self.sub_list = sub_list
		self.viz_type = viz_type

		self.root = master
		self.v = tk.IntVar()
		self.v.set(3)

		self.save_var = tk.IntVar()
		self.save_var.set(0)

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

		save_button_frame = tk.Frame(self.subject_window)
		tk.Checkbutton(save_button_frame, text="Save Figure", variable=self.save_var).pack()
		save_button_frame.pack(side="bottom")


	def button_click(self, stim, stim_num=-1):
		if self.v.get() == 1:
			stim.visualize()

		elif self.v.get() == 2:
			if self.save_var.get() == 1:
				stim.gazePlot(save_fig=True)
			else:
				stim.gazePlot()

		elif self.v.get() == 3:
			if self.viz_type == "group":
				stim_name = {stim.stim_type : stim_num}
				# self.subject_window.destroy()
				if self.save_var.get() == 1:
					groupHeatMap(self.sub_list, stim_name, self.json_file, save_fig=True)
				else:
					groupHeatMap(self.sub_list, stim_name, self.json_file)

			else:
				if self.save_var.get() == 1:
					stim.gazeHeatMap(save_fig=True)
				else:
					stim.gazeHeatMap()


class Subject:
	"""This class deals with encapsulating all the relevant information regarding a subject who participated in an experiment. The class contains functions that help in the extraction of data from the databases (SQL or CSV) and the creation of the `Stimuli <#module-Stimulus>`_ objects. The class also calculates the control data for the purpose of standardisation.

	Parameters
	---------
	name: str
		Name of the Subject
	subj_type: str
		Type of the Subject
	stimuli_names: list(str)
		List of stimuli that are to be considered for extraction
	columns: list(str)
		List of columns that need to be extracted from the database
	json_file: str
		Name of json file that contains information regarding the experiment/database
	sensors: list(str)
		Contains the names of the different sensors whose indicators are being analysed
	database: str | SQL object
		is the SQL object for the SQL database | name of the folder that contains the name csv files
	manual_eeg: bool
		Indicates whether artifact removal is manually done or not
	reading_method: str
		Mentions the format in which the data is being stored

	"""

	def __init__(self, path, name, subj_type, stimuli_names, columns, json_file, sensors, database, reading_method, aoi):
		a = datetime.now()
		self.path = path
		self.sensors = sensors
		self.stimuli_names = stimuli_names
		self.name = name
		self.subj_type = subj_type
		self.json_file = json_file
		self.aoi = aoi
		self.stimulus = self.stimulusDictInitialisation(stimuli_names, columns, json_file, sensors, database, reading_method)
		self.control_data = self.getControlData()
		self.aggregate_meta = {}
		b = datetime.now()
		print("Total time to instantiate subject ", self.name, ": ", (b-a).seconds, "s\n")


	def dataExtraction(self, columns, json_file, database, reading_method, stimuli_names):
		"""Extracts the required columns from the data base and the required stimuli_column and returns a pandas datastructure

		Parameters
		----------
		columns: list(str)
			list of the names of the columns of interest
		json_file: str
			Name of the json file that contains information about the experiment
		database: SQL object| str
			is the SQL object for the SQL database | name of the folder that contains the name csv files
		reading_method: str {"SQL","CSV"}
			Describes which type of databse is to be used for data extraction
		stimuli_names: list(str)
			List of stimuli that are to be considered for extraction

		Returns
		-------
		df: pandas DataFrame
			contains the data of columns of our interest

		"""

		if reading_method == "SQL":

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

			#NTBD: Change StimulusName from being Hardcoded
			query = string + ' FROM "' + self.name + '" WHERE StimulusName in ('
			flag = -1

			for k in stimuli_names:
				for name in stimuli_names[k]:

					if flag == -1:
						flag = 0
						selected_stimuli = "'" + name + "'"
					else:
						selected_stimuli = selected_stimuli + ", '" + name + "'"

			query = query + selected_stimuli + ")"

			dummy = database.execute(query)

			conversion = pd.DataFrame(dummy.fetchall())
			conversion.columns = dummy.keys()

			return conversion

		elif reading_method == "CSV":

			a = datetime.now()
			column_names = []

			for name in columns:
				column_names.append(name)

			csv_file = database + "/" + self.name + ".csv"
			df = pd.read_csv(csv_file, usecols = column_names)
			df = df.replace(to_replace=r'Unnamed:*', value=float(-1), regex=True)
			b = datetime.now()
			# print("Query: ", (b-a).seconds)

			return df

		else:
			print("Neither of the 2 options have been chosen")
			return None


	def timeIndexInitialisation(self, stimulus_column_name, stimulus_name, df):
		"""This function that will retireve the index of the start, end and roi of a question

		Parameters
		----------
		stimulus_column_name: str
			Name of the column where the stimuli names are present
		stimulus_name: str
			Name of the stimulus
		df: pandas dataframe
			Contains the data from which `start`, `end` and `roi` will be extracted from

		Returns
		-------
		start: int
			The index of the start of a queation
		end: int
			The index of the end of a question
		roi: int
			The index when the eye lands on the region of interest

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

		return start,end,roi


	def stimulusDictInitialisation(self, stimuli_names, columns, json_file, sensors, database, reading_method):
		"""Creates a list of objects of class `Stimuli <#module-Stimulus>`_.

		Parameters
		---------
		stimuli_names: list(str)
			list of names of different stimulus
		columns: list(str)
			list of names of the columns of interest
		json_file: str
			Name of json file that contains information about the experiment/database
		sensors: object of class Sensor
			Is an object of class sensor and is used to see if EEG extraction is required
		database: SQL object | str
			Is the SQL object that is created for accessing the SQL database | Name of the folder containing the CSV files
		reading_method: str {"SQL","CSV"}
			Describes which type of databse is to be used for data extraction

		Returns
		-------
		stimulus_object_dict: dict
			dictionary of objects of class stimulus ordered by category

		"""

		data = self.dataExtraction(columns,json_file, database, reading_method, stimuli_names)

		stimulus_object_dict = {}

		for category in stimuli_names:

			stimulus_object_list = []

			for stimulus_name in stimuli_names[category]:
				#NTBD change the harcoding of the stimulusName
				start_time, end_time, roi_time = self.timeIndexInitialisation("StimulusName",stimulus_name, data)

				stimuli_data = data[start_time : end_time+1]

				stimulus_object = Stimulus(self.path, stimulus_name, category, sensors, stimuli_data, start_time, end_time, roi_time, json_file, self.name, self.aoi)

				stimulus_object_list.append(stimulus_object)

			stimulus_object_dict[category] = stimulus_object_list

		return stimulus_object_dict


	def getControlData(self):
		"""Function to find the values for standardization/normalization of the features extracte from the different stimuli.

		The method is invoked implicitly by the `__init__` method. It extracts the features/metadata for the stimuli mentioned in the json file under the "*Control_Questions*" field. If it is blank the values in the control data structure will be all 0s. During analysis, these control values will be subtracted from the values found for each stimulus.

		Returns
		-------
		control : dict
			Dictionary containing the standardised values for all the metadata/features. The keys of the dictionary are the different meta columns for a given sensor type. It can be found under meta_cols in `Sensor <#module-Sensor>`_.

		"""

		control = dict()
		for sen in self.sensors:
			control.update({sen:dict()})
			for meta in Sensor.meta_cols[sen]:
				control[sen].update({meta: 0})

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		if "Control_Questions" in json_data:
			if not os.path.isdir(self.path + '/control_values/'):
				os.makedirs(self.path + '/control_values/')

			if os.path.isfile(self.path + '/control_values/' + self.name + '.pickle') == True:
				pickle_in = open(self.path + '/control_values/' + self.name + '.pickle',"rb")
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
				pickle_out = open(self.path + '/control_values/' + self.name + '.pickle',"wb")
				pickle.dump(control, pickle_out)
				pickle_out.close()

		return control


	def subjectVisualize(self, master, viz_type="individual", sub_list=None):
		"""Visualization function to open the window for stimulus and plot type selection.

		It is invoked internally by the `visualizaData <#Experiment.Experiment.visualizeData>`_ function.

		"""
		_ = SubjectVisualize(master, self.name, self.stimulus, json_file=self.json_file, viz_type=viz_type, sub_list=sub_list)


	def subjectAnalysis(self, average_flag, standardise_flag):
		"""Function to find features for all stimuli for a given subject.

		Does not return any value. It stores the calculated features/metadata in its `aggregate_meta` member variable. Can be accessed by an object of the class. For structure of this variable see `Subject <#module-Subject>`_.

		Parameters
		----------
		average_flag : bool
			If ``True``, average values for a given meta variable under each stimulus type for a given stimulus.
		standardise_flag : bool
			If ``True``, subtract `control_data` values for a given meta variable for each stimulus. See `getControlData <#Subject.Subject.getControlData>`_ for more details on `control_data`.

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
							if standardise_flag:
								self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], (np.array(stim.sensors["EyeTracker"].metadata[cd]) - self.control_data[sen][cd])))
							else:
								self.aggregate_meta[s][cd] = np.hstack((self.aggregate_meta[s][cd], stim.sensors["EyeTracker"].metadata[cd]))

		if average_flag:
			for s in self.stimulus:
				for sen in self.sensors:
					for cd in Sensor.meta_cols[sen]:
						self.aggregate_meta[s][cd] = np.array([np.mean(self.aggregate_meta[s][cd], axis=0)])