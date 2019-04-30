# -*- coding: utf-8 -*-

from Sensor import Sensor
from Subject import Subject
import numpy as np
import os
import time
import csv
from datetime import datetime
from scipy import stats
import json
import random
import pandas as pd
from sqlalchemy import create_engine
import pingouin as pg
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import tkinter as tk
from functools import partial
from stat_functions import *
from statistics import mean
import statsmodels.api as sm
import pickle


class Visualize:
	
	def __init__(self, master, subjects, exp):
		self.exp = exp
		master.title("PyTrack Visualization")
		self.root = master
		self.v = tk.IntVar()
		# self.v.set(1)
		self.subjects = subjects

		self.master_frame = tk.Frame(self.root, height=30, width=70)
		self.sub_frame = tk.Frame(self.master_frame, height=30, width=70)
		self.sub_frame.grid_propagate(False)
		self.text = tk.Text(self.sub_frame, width=50)
		
		self.submit_frame = tk.Frame(self.root, width=70)
		func = partial(self.button_click)
		self.submit_btn = tk.Button(self.submit_frame, text="Visualize", command=func)
		self.submit_btn.pack()

		viz_type_frame = tk.Frame(self.root, width=70)
		tk.Radiobutton(viz_type_frame, text="Group Visualization", padx=20, variable=self.v, value=1, command=self.subFrameSetup).pack(side="left", anchor=tk.W)
		tk.Radiobutton(viz_type_frame, text="Individual Visualization", padx=20, variable=self.v, value=2, command=self.subFrameSetup).pack(side="right", anchor=tk.W)
		viz_type_frame.pack(side="top", fill="both", expand=True)
		
		self.master_frame.pack()
		# self.subFrameSetup()

	def subFrameSetup(self):
		
		self.sub_frame.destroy()
		self.submit_frame.pack_forget()

		if self.v.get() == 1:
			
			self.sub_frame = tk.Frame(self.master_frame, height=30, width=70)
			self.sub_frame.grid_propagate(False)
			self.text = tk.Text(self.sub_frame, width=50)

			self.chk_bt_var = [tk.IntVar() for i in range(len(self.subjects))]

			for i, sub in enumerate(self.subjects):
				func = partial(self.button_click, sub)
				chk_bt = tk.Checkbutton(self.sub_frame, text=sub.name, variable=self.chk_bt_var[i], onvalue=1, offvalue=0)
				self.text.window_create(tk.END, window=chk_bt)
				self.text.insert(tk.END, "\n")	
			
			self.submit_frame.pack(side="bottom", fill="both", expand=True)

		else:
			
			self.sub_frame = tk.Frame(self.master_frame, height=30, width=70)
			self.sub_frame.grid_propagate(False)
			self.text = tk.Text(self.sub_frame, width=50)

			for i, sub in enumerate(self.subjects):
				func = partial(self.button_click, sub)
				bt = tk.Button(self.sub_frame, width=30, text=sub.name, command=func)
				self.text.window_create(tk.END, window=bt)
				self.text.insert(tk.END, "\n")	

		vsb = tk.Scrollbar(self.sub_frame, orient="vertical")
		vsb.config(command=self.text.yview)
		
		self.text.configure(yscrollcommand=vsb.set)
		
		self.text.pack(side="left", fill="both", expand=True)
		vsb.pack(side="right", fill="y")

		self.sub_frame.pack(side="bottom", fill="both", expand=True)

	def button_click(self, sub=None):
		if sub == None:
			sub_list = []
			for ind, i in enumerate(self.chk_bt_var):
				if i.get() == 1:
					sub_list.append(self.subjects[ind])
			
			sub_list[0].subjectVisualize(self.root, viz_type="group", sub_list=sub_list)

		else:
			sub.subjectVisualize(self.root)


class Experiment:
	""" This is the main class that performs statistical analysis of the data contained in a database

	Parameters
	-----------
	name: str
		Name of the experiment
	json_file: str
		Name of the json file that contains information regarding the experiment or the database
	sensors: list(str)
		Contains the names of the different sensors whose indicators are being analysed
	reading_method: str {"SQL"| "CSV"}
		Mentions the format in which the data is being stored
	"""


	def __init__(self, json_file, reading_method, sensors=["EyeTracker"]):

		with open(json_file, "r") as json_f:
			json_data = json.load(json_f)

		self.path = json_data["Path"]
		self.name = json_data["Experiment_name"]
		self.json_file = json_file #string
		self.sensors = sensors
		self.columns = self.columnsArrayInitialisation()
		self.stimuli = self.stimuliArrayInitialisation() #dict of names of stimuli demarcated by category
		self.subjects = self.subjectArrayInitialisation(reading_method) #list of subject objects
		self.meta_matrix_dict = (np.ndarray(len(self.subjects), dtype=str), dict())
		

		if not os.path.isdir(self.path + '/Subjects/'):
			os.makedirs(self.path + '/Subjects/')
		

	def stimuliArrayInitialisation(self):
		"""This functions instantiates the dictionary 'stimuli' with the list of names of the different stimuli by category

		Parameters
		----------
		json_file : str
			Name of the json file which contains details of the experiment

		Returns
		-------
		data_dict : dict
			Dictionary containing the names of the different stimuli categorised by type

		"""

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		stimuli_data = json_data["Stimuli"]

		data_dict = {}

		for k in stimuli_data:
			data_dict[k] = stimuli_data[k]

		return data_dict


	def subjectArrayInitialisation(self, reading_method):
		"""This function initialises an list of objects of class Subject

		Parameters
		----------
		reading_method: str {'SQL','CSV'}, optional
			Specifies the database from which data extraction is to be done from	

		Returns
		-------
		subject_list : list(Subject objects)
			list of objects of class Subject
		
		"""

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		subject_list = []	

		subject_data = json_data["Subjects"]

		if reading_method == "SQL":
			name_of_database = json_data["Experiment_name"]
			path = self.path + "/" + name_of_database
			extended_name = "sqlite:///" + path + ".db"
			database = create_engine(extended_name)
		
		elif reading_method == "CSV":
			database = self.path + "/Data/csv_files/"

		for k in subject_data:

			for subject_name in subject_data[k]:

				subject_object = Subject(self.path, subject_name, k, self.stimuli, self.columns, self.json_file, self.sensors, database, reading_method)

				subject_list.append(subject_object)

		database.dispose()

		return subject_list


	def columnsArrayInitialisation(self):
		"""The functions extracts the names of the columns that are to analysed

		Parameters
		----------
		json_file: str
			Name of the json file which contains details of the experiment

		Returns
		-------
		columns_list: list(str)
			List of names of columns of interest
		
		"""

		with open(self.json_file) as json_f:
			json_data = json.load(json_f)

		column_list = []

		column_classes = [sen for sen in self.sensors]
		column_classes.append("Extra")

		for col_class in column_classes:
			for name in json_data["Columns_of_interest"][col_class]:
				column_list.append(name)

		return column_list


	def visualizeData(self):
		"""Function to open up the GUI for visualizing the data of the experiment.

		This function can be invoked by an `Experiment <#module-Experiment>`_ object. It opens up a window and allows the usee to visualize data such as dynamic gaze and pupil plots, fixation plots and gaze heat maps for individual subjects or aggregate heat maps for a group of subjects on a given stimulus.
		
		"""

		root = tk.Tk()
		root.resizable(False, False)
		viz = Visualize(root, self.subjects, self)
		root.mainloop()

	
	def analyse(self, standardise_flag=False, average_flag=False, parameter_list={"all"}, between_factor_list=["Subject_type"], within_factor_list=["Stimuli_type"], statistical_test="Mixed_anova"):
		"""This function carries out the required statistical analysis technique for the specified indicators/parameters

		Parameters
		----------
		standardise_flag: bool {``False``, ``True``}
			Indicates whether the data being considered need to be standardised (by subtracting the control values/baseline value) 
		average_flag: bool {``False``, ``True``} 
			Indicates whether the data being considered should averaged across all stimuli of the same type
		parameter_list: set {{"all"}}
			Set of the different indicators/parameters (Pupil_size, Blink_rate) on which statistical analysis is to be performed 
		between_factor_list: list(str) {["Subject_type"]} 
			List of between group factors
		within_factor_list: list(str) {["Stimuli_type"]} 
			List of within group factors
		statistical_test: str {"Mixed_anova","RM_anova","ttest"}
			Name of the statistical test that has to be performed
		
		"""
		
		#Defining the meta_matrix_dict data structure
		for sensor_type in Sensor.meta_cols:
			for meta_col in Sensor.meta_cols[sensor_type]:
				self.meta_matrix_dict[1].update({meta_col : np.ndarray((len(self.subjects), len(self.stimuli)), dtype=object)})

		#Instantiation of the meta_matrix_dict database		
		for sub_index, sub in enumerate(self.subjects):
			sub.subjectAnalysis(average_flag, standardise_flag)

			self.meta_matrix_dict[0][sub_index] = sub.subj_type

			for stim_index, stimuli_type in enumerate(sub.aggregate_meta):

				for meta in sub.aggregate_meta[stimuli_type]:
					self.meta_matrix_dict[1][meta][sub_index, stim_index] = sub.aggregate_meta[stimuli_type][meta]

		#To find the number of subjects in Group 1 (Innocent) and Group 2 (Guilty)
		with open(self.json_file, "r") as json_f:
			json_data = json.load(json_f)
			num_inn = len(json_data["Subjects"]["innocent"])
			num_guil = len(json_data["Subjects"]["guilty"])
		
		num_samples = 2 #Number of participants randomly picked from each group
		num_runs = 1 #Number of times the experiment anlaysis is run (Results may differ in each run as the participants chosen will be different)


		for sen in self.sensors: #For each type of sensor
			for i in range(num_runs): #For each run

				#sub_indices is a list of indices, where each index refers to a subject
				sub_indices = np.hstack((random.sample(range(0, num_inn), num_samples), random.sample(range(num_inn, num_inn + num_guil), num_samples)))
				print("The indices chosen are:", sub_indices)

				head_row = ["Indices"]
				csv_row = [str(sub_indices)]

				with open(self.json_file) as json_f:
					json_data = json.load(json_f)

				for meta in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
					if meta == "pupil_size" or meta == "pupil_size_downsample" or meta == "sacc_count" or meta == "sacc_duration" or meta == "sacc_vel" or meta == "sacc_amplitude":
						continue

					if 'all' not in parameter_list:

						if meta not in parameter_list:
							print(meta)
							continue

					print("\n\n")
					print("\t\t\t\tAnalysis for ",meta)	

					#For the purpose of statistical analysis, a pandas dataframe needs to be created that can be fed into the statistical functions
					#The columns required are - meta (indicator), the between factors (eg: Subject type or Gender), the within group factor (eg: Stimuli Type), Subject name/id

					#Defining the list of columns required for the statistical analysis
					column_list = [meta]

					column_list.extend(between_factor_list)
					column_list.extend(within_factor_list)
					column_list.append("subject")

					data =  pd.DataFrame(columns=column_list)


					#For each subject
					for sub_index in sub_indices:
						
						sub = self.subjects[sub_index] #sub refers to the object of the respective subject whose data is being extracted

						#For each Question Type (NTBC: FIND OUT WHAT THE AGGREGATE_META CONTAINS)
						for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):

							#NTBC: Ignore the picture stimuli for the time being
							if stimuli_type not in ['alpha', 'general', 'general_lie', 'relevant']:
								continue

							#Value is an array (NTBC : Is it always an array or can it also be a single value?)	
							value_array = self.meta_matrix_dict[1][meta][sub_index,stimuli_index]
							print(len(value_array))

							try:					
								for value in value_array:

									row = []

									row.append(value)
									row.append(sub.subj_type)

									#Add the between group factors (need to be defined in the json file)
									for param in between_factor_list:

										if param == "Subject_type":
											continue

										try:
											row.append(json_data["Subjects"][sub.subj_type][sub.name][param])
										except:
											print("Between subject paramter: ", param, " not defined in the json file")	

									#NTBD: Please change sub.name to required value
									#DO NOT HAVE ACCESS TO THE NAME OF THE STIMULI, SO NEED TO FIGURE OUT HOW THE WITHIN GROUP 
									#PARAMETERS NEED TO BE ACCESSED

									row.append(stimuli_type)

									for param in within_factor_list:
										
										if param == "Stimuli_type":
											continue

										try:
											stimulus_name = self.stimuli[stimuli_type][stimuli_index]
											row.append(json_data["Stimuli"][stimuli_type][stimuli_name][param])
										except:
											print("Within stimuli parameter: ", param, " not defined in the json file")

									row.append(sub.name)

									#NTBC: Checking condition if value is nan for error checking
									if(np.isnan(value)):
										print(row)

									#Instantiate into the pandas dataframe
									data.loc[len(data)] = row

							except:
								print("Error in data instantiation")

					#Depending on the parameter, choose the statistical test to be done

					if(statistical_test == "Mixed_anova"):

						print(meta, "Mixed anova")
						mixed_anova_calculation(meta, data, between_factor_list[0], within_factor_list[0])

					elif(statistical_test == "RM_anova"):

						print(meta, "RM Anova")
						rm_anova_calculation(meta, data, within_factor_list)

					elif(statistical_test == "ttest"):

						print(meta, "t test")
						ttest_calculation(meta, data, between_factor_list, within_factor_list)

					#4. NTBD : Genlie vs Relevant comparison to be done	


	def logistic_regression_analysis(self, average_flag=False, standardise_flag=False,  independent_parameter_list=["all"]):
		"""Run a logistic regression on the data from several parameters

		Parameters
		----------
		standardise_flag: bool {``False``, ``True``}
			Indicates whether the data being considered need to be standardised (by subtracting the control values/baseline value) 		
		average_flag: bool {``False``, ``True``} 
			Indicates whether the data being considered should averaged across all stimuli of the same type
		independent_parameter_list = list of strings {["all"]}
			Is a list of the independent variables in the logistic regression equation

		"""

		#Defining the meta_matrix_dict data structure
		for sensor_type in Sensor.meta_cols:
			for meta_col in sensor_type:
				self.meta_matrix_dict[1].update({meta_col : np.ndarray((len(self.subjects), len(self.stimuli)), dtype=object)})

		#Instantiation of the meta_matrix_dict database		
		for sub_index, sub in enumerate(self.subjects):
			sub.subjectAnalysis(average_flag, standardise_flag)

			self.meta_matrix_dict[0][sub_index] = sub.subj_type

			for stim_index, stimuli_type in enumerate(sub.aggregate_meta):
				for meta in sub.aggregate_meta[stimuli_type]:
					self.meta_matrix_dict[1][meta][sub_index, stim_index] = sub.aggregate_meta[stimuli_type][meta]

		#Define a pandas dataframe with the required column names
		#NTBD add stimuli_type as a value as well 
		#(see if pingouin takes nominal factors as well, if not search for a package that does take nominal values for independent factors)

		if independent_parameter_list.count("all") == 1:

			independent_parameter_list = []

			for sen in self.sensors:
				for meta in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
					independent_parameter_list.append(meta)


		dataframe_columns = ["Subject_type"]

		dataframe_columns.extend(independent_parameter_list)

		dataframe_columns.append("Stimuli_type")

		data =  pd.DataFrame(columns=dataframe_columns)


		#To find the number of subjects in Group 1 (Innocent) and Group 2 (Guilty)
		with open(self.json_file, "r") as json_f:
			json_data = json.load(json_f)
			num_inn = len(json_data["Subjects"]["innocent"])
			num_guil = len(json_data["Subjects"]["guilty"])
		
		num_samples = 5 #Number of participants randomly picked from each group
		num_runs = 1 #Number of times the experiment anlaysis is run (Results may differ in each run as the participants chosen will be different)

		sub_indices = np.hstack((random.sample(range(0, num_inn), num_samples), random.sample(range(num_inn, num_inn + num_guil), num_samples)))
		
		for sub_index in sub_indices:
							
			sub = self.subjects[sub_index] #sub refers to the object of the respective subject whose data is being extracted

			#For each Question Type
			for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):

				if stimuli_type not in ['alpha', 'general', 'general_lie', 'relevant']:
					continue

				row = []

				if(sub.subj_type == "guilty"):
					row.append(1)
				elif(sub.subj_type == "innocent"):
					row.append(0)
				else:
					print("Unrecognized Subject type: ", sub.subj_type)		

				#NTBD: how to handle if all the parameters are required?
				
				for meta in independent_parameter_list:

					#NTBD: Lets assumes that value_array is always a single value and not a list

					value_array = self.meta_matrix_dict[1][meta][sub_index,stimuli_index]
					
					row.append(mean(value_array))

				row.append(stimuli_index)

				data.loc[len(data)] = row	

		#Convert the Stimuli_type into dummy variables

		stimuli_dummy_data = pd.get_dummies(data["Stimuli_type"])

		stimuli_columns = ['alpha', 'general', 'general_lie', 'relevant']
		stimuli_dummy_data.columns = stimuli_columns

		data = data.join(stimuli_dummy_data)

		print(independent_parameter_list)

		independent_parameter_list.extend(stimuli_columns)

		#Instantiate X and y values of the linear regression

		print(independent_parameter_list)

		X = data[independent_parameter_list]
		y = data["Subject_type"]

		dependent_column = []

		for i in range(len(y)):

			if(y[i] == 0.0):
				 dependent_column.append(0)
			elif(y[i] == 1.0):
				dependent_column.append(1)

		print(dependent_column)

		#Logistic regression through pingoiun

		'''

		print("\t\t\t\tPingouin logistic regression")


		results = pg.logistic_regression(X, dependent_column)

		print(results.round(2))

		'''

		#Logistic regression through statsmodel

		print("\t\t\t\tStatmodels logistic regression")

		logit = sm.Logit(dependent_column, X, fit_intercept = False)

		result = logit.fit()

		print(result.summary2())
	
	
	def getMetaData(self, sub, stim=None, sensor="EyeTracker"):
		"""Function to return the extracted features for a given subject/participant.

		Parameters
		----------
		sub : str
			Name of the subject/participant.
		stim : str | ``None``
			Name of the stimulus. If 'str', the features of the given stimulus will be returned. If ``None``, the features of all stimuli averaged for the different stimuli types (as mentoned in json file) is wanted. 
		sensor : str
			Name of the sensor for which the features is wanted.
		
		Returns
		-------
		dict
			
		Note
		----
		- If the `stim` is ``None``, the returned dictionary is organised as follows 
			{"Stim_TypeA": {"meta1":[], "meta2":[], ...}, "Stim_TypeB": {"meta1":[], "meta2":[], ...}, ...}
		- If the `stim` is ``str``, the returned dictionary is organised as follows 
			{"meta1":[], "meta2":[], ...}
			
		To get the names of all the metadata/features extracted, look at the `Sensor <#module-Sensor>`_ class

		"""
		if stim == None:
			sub_ind = self.subjects.index(sub)
			return self.subjects[sub_ind].aggregate_meta
		
		else:
			sub_ind = self.subjects.index(sub)
			stim_cat = ""
			stim_ind = -1
			for cat in self.stimuli:
				stim_ind = self.stimuli[cat].index(stim)
				if stim_ind != -1:
					stim_cat = cat
					break
			
			return self.subjects[sub_ind].stimulus[stim_cat][stim_ind].sensors[sensor].metadata