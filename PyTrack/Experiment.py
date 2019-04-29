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

	def __init__(self, json_file, sensors=["EyeTracker"]):

		with open(json_file, "r") as json_f:
			json_data = json.load(json_f)

		self.path = json_data["Path"]
		self.name = json_data["Experiment_name"]
		self.json_file = json_file #string
		self.sensors = sensors
		self.columns = self.columnsArrayInitialisation()
		self.stimuli = self.stimuliArrayInitialisation() #dict of names of stimuli demarcated by category
		self.subjects = self.subjectArrayInitialisation() #list of subject objects
		self.meta_matrix_dict = (np.ndarray(len(self.subjects), dtype=str), dict())
		

		if not os.path.isdir('./Subjects/'):
			os.makedirs('./Subjects/')
		

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

		name_of_database = json_data["Experiment_name"]
		extended_name = "sqlite:///" + name_of_database + ".db"
		database = create_engine(extended_name)

		for k in subject_data:

			for subject_name in subject_data[k]:

				subject_object = Subject(self.path, subject_name, k, self.stimuli, self.columns, self.json_file, self.sensors, database)

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

	
	def analyse(self, average_flag=True, standardise_flag=False, stat_test=True):
		
		for sensor_type in Sensor.meta_cols:
			for meta_col in Sensor.meta_cols[sensor_type]:
				self.meta_matrix_dict[1].update({meta_col : np.ndarray((len(self.subjects), len(self.stimuli)), dtype=object)})

		for sub_index, sub in enumerate(self.subjects):
			sub.subjectAnalysis(average_flag, standardise_flag)

			self.meta_matrix_dict[0][sub_index] = sub.subj_type

			for stim_index, stimuli_type in enumerate(sub.aggregate_meta):
				for meta in sub.aggregate_meta[stimuli_type]:
					self.meta_matrix_dict[1][meta][sub_index, stim_index] = sub.aggregate_meta[stimuli_type][meta]

		if stat_test:
			#For each column parameter
			p_value_table = pd.DataFrame()
			flag = 1

			with open(self.json_file, "r") as json_f:
				json_data = json.load(json_f)
				num_inn = len(json_data["Subjects"]["innocent"])
				num_guil = len(json_data["Subjects"]["guilty"])
			num_samples = 7

			for sen in self.sensors:
				for i in range(10):

					sub_indices = np.hstack((random.sample(range(0, num_inn), num_samples), random.sample(range(num_inn, num_inn + num_guil), num_samples)))
					print(sub_indices)

					# sub_indices = np.hstack((random.sample(range(0, 2), 2), random.sample(range(2, 4), 2)))
					# print(sub_indices)

					for meta in Sensor.meta_cols[Sensor.sensor_names.index(sen)]:
						if meta == "pupil_size" or meta == "pupil_size_downsample":
							continue

						print("\n\n")
						print("\t\t\t\tAnalysis for ",meta)	
						data =  pd.DataFrame(columns=[meta,"stimuli_type","individual_type","subject"])

						#For each subject
						for sub_index in sub_indices:
							
							sub = self.subjects[sub_index]
							#For each Question Type
							for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):
								if stimuli_type not in ['alpha', 'general', 'general_lie', 'relevant']:
									continue
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
									print("Value array for ", stimuli_type, " is empty")
							
						#print(data,"\n\n")

						column_values = []

						aov = pg.mixed_anova(dv=meta, within='stimuli_type', between='individual_type', subject = 'subject', data=data)
						posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', between='individual_type', subject='subject', data=data)

						if(flag == 1):
							
							flag = 0
							column_values.append(aov['Source'][0] + ' - ANOVA')
							column_values.append(aov['Source'][2] + ' - ANOVA')

							contrast = posthocs['Contrast'][6:11]
							stimuli = posthocs['stimuli_type'][6:11]

							for i in range(6,11):
								column_values.append(posthocs["Contrast"][i] + ' ' + posthocs["stimuli_type"][i])

							column_values.append("innocent_genlie_relevant")
							column_values.append("guilty_genlie_relevant")
								
							p_value_table['Row_names'] = column_values


						column_values = []
						
						# Pretty printing of ANOVA summary
						#pg.print_table(aov)

						column_values.append(aov['p-unc'][0])
						column_values.append(aov['p-unc'][2])

						#pg.print_table(posthocs)

						p_values_ttest = posthocs['p-unc'][6:11]
						for value in p_values_ttest:
							column_values.append(value)

						'''
						if meta == "response_time":

							scipy.stats.ttest_ind(a, b)
							scipy.stats.ttest_ind(a, b, equal_var=False)
						'''
						#t-test comparison of general lie and relevant for innocent and guilty participants

						innocent_data = data.loc[(data['individual_type'] == 'innocent') & ((data['stimuli_type'] == 'relevant') | (data['stimuli_type'] == 'general_lie'))]

						aov = pg.rm_anova(dv=meta, within='stimuli_type', subject = 'subject', data=innocent_data)
						posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', subject='subject', data=innocent_data)

						#print(pg.print_table(aov))
						#print(pg.print_table(posthocs))

						column_values.append(posthocs['p-unc'][0])

						guilty_data = data.loc[(data['individual_type'] == 'guilty') & ((data['stimuli_type'] == 'relevant') | (data['stimuli_type'] == 'general_lie'))]

						aov = pg.rm_anova(dv=meta, within='stimuli_type', subject = 'subject', data=guilty_data)
						posthocs = pg.pairwise_ttests(dv=meta, within='stimuli_type', subject='subject', data=guilty_data)

						#print(pg.print_table(aov))
						#print(pg.print_table(posthocs))

						column_values.append(posthocs['p-unc'][0])

						p_value_table[meta] = column_values

					p_value_table.to_csv("p_values"+ str(i) + ".csv" , index = False)


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