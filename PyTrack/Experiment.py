# -*- coding: utf-8 -*-

import os
import json
import tkinter as tk
from functools import partial
import csv

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from statsmodels.formula.api import ols
from matplotlib.widgets import RectangleSelector, PolygonSelector, EllipseSelector

from PyTrack.Sensor import Sensor
from PyTrack.Subject import Subject


class Visualize:

	def __init__(self, master, subjects, exp):
		self.exp = exp
		master.title("PyTrack Visualization")
		self.root = master
		self.v = tk.IntVar()
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
	""" This is class responsible for the analysis of data of an entire experiment. The creation of a an object of this class will subsequently create create objects for each `Subject <#module-Subject>`_. involved in the experiment (which in turn would create object for each `Stimulus <#module-Stimulus>`_ which is viewed by the subject).

	This class also contains the `analyse <#Experiment.Experiment.analyse>`_ function which is used for the statistical analysis of the data (eg: Mixed ANOVA, RM ANOVA etc).

	Parameters
	-----------
	json_file: str
		Name of the json file that contains information regarding the experiment or the database
	reading_method: str {"SQL", "CSV"} (optional)
		Mentions the format in which the data is being stored
	sensors: list(str) (optional)
		Contains the names of the different sensors whose indicators are being analysed (currently only Eye Tracking can be done
		However in future versions, analysis of ECG and EDA may be added)
	aoi : str {'NA', 'e', 'r', 'p'} | tuple (optional)
		If 'NA' then AOI is the entire display size. If 'e' then draw ellipse, 'p' draw polygon and 'r' draw rectangle. If type is ``tuple``, user must specify the coordinates of AOI in the following order (start_x, start_y, end_x, end_y). Here, x is the horizontal axis and y is the vertical axis.

	"""

	def __init__(self, json_file, reading_method="SQL", aoi="NA"):

		json_file = json_file.replace("\\", "/")

		with open(json_file, "r") as json_f:
			json_data = json.load(json_f)

		json_data["Path"] = json_data["Path"].replace("\\", "/")

		self.path = json_data["Path"]
		self.name = json_data["Experiment_name"]
		self.json_file = json_file #string
		self.sensors = ["EyeTracker"]
		self.aoi = aoi
		self.aoi_coords = None
		# Setting AOI coordinates
		if isinstance(aoi, str):
			if aoi != "NA":
				self.aoi_coords = self.drawAOI()
			else:
				self.aoi_coords = [0.0, 0.0, float(json_data["Analysis_Params"]["EyeTracker"]["Display_width"]), float(json_data["Analysis_Params"]["EyeTracker"]["Display_height"])]
		else:
			self.aoi_coords = aoi

		json_data["Analysis_Params"]["EyeTracker"]["aoi"] = self.aoi_coords

		with open(self.json_file, "w") as f:
			json.dump(json_data, f, indent=4)

		self.columns = self.columnsArrayInitialisation()
		self.stimuli = self.stimuliArrayInitialisation() #dict of names of stimuli demarcated by category
		self.subjects = self.subjectArrayInitialisation(reading_method) #list of subject objects
		self.meta_matrix_dict = (np.ndarray(len(self.subjects), dtype=str), dict())

		if not os.path.isdir(self.path + '/Subjects/'):
			os.makedirs(self.path + '/Subjects/')


	def stimuliArrayInitialisation(self):
		"""This functions instantiates the dictionary `stimuli` with the list of names of the different stimuli by category

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

			if isinstance(stimuli_data[k], list):
				data_dict[k] = stimuli_data[k]

			elif isinstance(stimuli_data[k], dict):
				#Need to create a list of names of the stimuli
				list_stimuli = []
				for _, value in enumerate(stimuli_data[k]):
					list_stimuli.append(value)

				data_dict[k] = list_stimuli
			else:
				print("The Stimuli subsection of the json file is not defined properly")

		return data_dict


	def subjectArrayInitialisation(self, reading_method):
		"""This function initialises an list of objects of class `Subject <#module-Subject>`_.

		Parameters
		----------
		reading_method: str {'SQL','CSV'}
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
			extended_name = "sqlite:///" + self.path + "/Data/" + name_of_database + ".db"
			database = create_engine(extended_name)

		elif reading_method == "CSV":
			database = self.path + "/Data/csv_files/"

		for k in subject_data:

			if isinstance(subject_data[k], list):
				for subject_name in subject_data[k]:
					subject_object = Subject(self.path, subject_name, k, self.stimuli, self.columns, self.json_file, self.sensors, database, reading_method, self.aoi_coords)
					subject_list.append(subject_object)

			elif isinstance(subject_data[k], dict) :
				for _, subject_name in enumerate(subject_data[k]):
					subject_object = Subject(self.path, subject_name, k, self.stimuli, self.columns, self.json_file, self.sensors, database, reading_method, self.aoi_coords)
					subject_list.append(subject_object)

			else:
				print("The Subject subsection of the json file is not defined properly")

		if reading_method == "SQL":
			database.dispose()

		return subject_list


	def columnsArrayInitialisation(self):
		"""The functions extracts the names of the columns that are to be analysed

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
		_ = Visualize(root, self.subjects, self)
		root.mainloop()


	def metaMatrixInitialisation(self, standardise_flag=False, average_flag=False):
		"""This function instantiates the ``meta_matrix_dict`` with values that it extracts from the ``aggregate_meta`` variable of each Subject object.

		Parameters
		----------
		standardise_flag: bool (optional)
			Indicates whether the data being considered need to be standardised (by subtracting the control values/baseline value)
		average_flag: bool (optional)
			Indicates whether the data being considered should averaged across all stimuli of the same type
			NOTE: Averaging will reduce variability and noise in the data, but will also reduce the quantum of data being fed into the statistical test

		"""

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


	def return_index(self, value_index, summation_array):
		"""This function is used in helping to find the corresponding stimuli for data points in certain parameters, that more than one value for a specific stimuli

		Parameters
		----------
		
		value_index: int 
			Index of an instance of the parameter in the value array of the meta_matrix_dict
		summation_array: list
			list of values whose index will indicate which stimuli an instance will correspond to

		Returns
		-------
		summation_index: int
			Is the index of the stimuli to which an instance of the parameter corresponds to

		"""

		summation_index = -1

		for i, _ in enumerate(summation_array):

			if summation_array[i+1] == -1:
				if value_index >= summation_array[i] and value_index < summation_array[i+2]:
					summation_index = i
					break
			else:
				if value_index >= summation_array[i] and value_index < summation_array[i+1]:
					summation_index = i
					break

		return summation_index

	def summationArrayCalculation(self, meta,sub_index,stimuli_index):
		"""This function is used for creating a list that will be used later for identifying the corresponding stimuli for an instance in the meta_matrix_dict

		Parameters
		----------
		meta: str
			Is the parameter that is being considered
		sub_index: int
			Is the index of subject with regard to meta_matrix_dict 
		sub_index: int
			Is the index of subject with regard to meta_matrix_dict 
		
		Returns
		-------
		summation_array: list
			list of values whose index will indicate which stimuli an instance will correspond to

		"""

		if meta in ["sacc_duration", "sacc_vel", "sacc_amplitude"]:
			meta_col = "sacc_count"
		elif meta in ["ms_duration", "ms_vel", "ms_amplitude"]:
			meta_col = "ms_count"

		value_array = self.meta_matrix_dict[1][meta_col][sub_index,stimuli_index]

		summation_array =  [-1]
		sum_value = 0

		for i, _ in enumerate(value_array):

			if value_array[i] == 0:
				summation_array.append(-1)
			else:
				sum_value = sum_value + value_array[i]
				summation_array.append(sum_value)

		summation_array.append(10000)

		return summation_array



	def fileWriting(self, writer, csvFile, pd_dataframe, values_list):
		"""This function is used to write the statistical results into a csv file

		Parameters
		----------

		writer: file object
			File object that is used to write into a csv file 
		csvFile: str
			Name of the csv file
		pd_dataframe: pandas DataFrame
			Statistical results
		values_list: list
			Used for labelling the results

		"""

		writer.writerow(values_list)
		writer.writerow("\n")
		pd_dataframe.to_csv(csvFile)
		writer.writerow("\n")


	def welch_ttest(self, dv, factor, subject, data):
		"""This funtion is used to calculate the welch ttest (used when unequal variance of 2 samples exists)

		Parameters
		----------
		dv: str
			Name of the parameter that is being considered for statistical analysis
		factor: str
			Name of the factor on which statistical analysis is being done
		subject: str
			Name of the subject
		data: pandas DataFrame
			Data on which the Welch t-test is to be performed
		
		Returns
		-------
		normality: pandas DataFrame
			Data regarding normality of the different categories of the 'factor' 
		results: pandas DataFrame
			Data containing the results of the Welch t-test

		"""

		#Find number of unique values in the factor

		list_values = data[factor].unique()

		column_results=["Factor1","Factor2","dof","t-stastistic","p-value"]
		results = pd.DataFrame(columns=column_results)

		column_normality=["Factor","W test statistic","p-value"]
		normality = pd.DataFrame(columns=column_normality)

		#Calculating the normality of different values
		for value in list_values:
			row =[value]
			x=data[data[factor] == value]
			x=x[dv]
			w,p =stats.shapiro(x)
			row.extend([w,p])
			normality.loc[len(normality)] = row

		#Find the pariwise ttest for all of them
		for i,_ in enumerate(list_values):
			for j,_ in enumerate(list_values):

				if(i<j):

					row =[list_values[i],list_values[j]]
					x=data[data[factor] == list_values[i]]
					x=x[dv]
					y=data[data[factor] == list_values[j]]
					y=y[dv]
					t,p = stats.ttest_ind(x,y, equal_var = False)
					dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
					row.extend([dof,t,p])
					results.loc[len(results)] = row

		return normality,results


	def analyse(self, parameter_list={"all"}, between_factor_list=["Subject_type"], within_factor_list=["Stimuli_type"], statistical_test="Mixed_anova", file_creation=True, ttest_type=1):
		"""This function carries out the required statistical analysis.

		 The analysis is carried out on the specified indicators/parameters using the data extracted from all the subjects that were mentioned in the json file. There are 4 different tests that can be run, namely - Mixed ANOVA, Repeated Measures ANOVA, T Test and Simple ANOVA (both 1 and 2 way)

		Parameters
		----------
		parameter_list: set (optional)
			Set of the different indicators/parameters (Pupil_size, Blink_rate) on which statistical analysis is to be performed, by default it will be "all" so that all the parameter are considered.
		between_factor_list: list(str) (optional)
			List of between group factors, by default it will only contain "Subject_type".
			If any additional parameter (eg: Gender) needs to be considered, then the list will be: between_factor_list = ["Subject_type", "Gender"].
			DO NOT FORGET TO INCLUDE "Subject_type", if you wish to consider "Subject_type" as a between group factor.
			Eg: between_factor_list = ["factor_x"] will no longer consider "Subject_type" as a factor.
			Please go through the README FILE to understand how the JSON FILE is to be written for between group factors to be considered.
		within_factor_list: list(str) (optional)
			List of within group factors, by default it will only contain "Stimuli_type"
			If any additional parameter, needs to be considered, then the list will be: between_factor_list = ["Subject_type", "factor_X"].
			DO NOT FORGET TO INCLUDE "Stimuli_type", if you wish to consider "Stimuli_type" as a within group factor.
			Eg: within_factor_list = ["factor_x"] will no longer consider "Stimuli_type" as a factor.
			Please go through how the README FILE to understand how the JSON FILE is to be written for within group factors to be considered.
		statistical_test: str {"Mixed_anova","RM_anova","ttest","anova","None"} (optional)
			Name of the statistical test that has to be performed.
				NOTE:

				- ttest: There are 3 options for ttest, and your choice of factors must comply with one of those options, for more information, please see description of `ttest_type` variable given below.
				- Welch_ttest: There are 2 options for Welch Ttest, and your choice of factors must comply with one of those options, for more information, please see description of `ttest_type` variable given below.
				- Mixed_anova: Only 1 between group factor and 1 within group factor can be considered at any point of time
				- anova: Any number of between group factors can be considered for analysis
				
				- RM_anova: Upto 2 within group factors can be considered at any point of time
		file_creation: bool (optional)
			Indicates whether a csv file containing the statistical results should be created.
				NOTE:
				The name of the csv file created will be by the name of the statistical test that has been chosen.
				A directory called "Results" will be created within the Directory whose path is mentioned in the json file and the csv files will be stored within "Results" directory.
				If any previous file by the same name exists, it will be overwritten.
		ttest_type: int {1,2,3} (optional)
			Indicates what type of parameters will be considered for the ttest and Welch Ttest
				NOTE:
				For ttest-

				- 1: Upto 2 between group factors will be considered for ttest
				- 2: 1 within group factor will be considered for ttest
				
				- 3: 1 within group and 1 between group factor will be considered for ttest

				For Welch ttest-

				- 1: Will consider the first factor in 'between_factor_list'

				- 2: Will consider the first factor in 'within_factor_list' 

		Examples
		--------

		For calculating Mixed ANOVA, on all the parameters, with standardisation, NOT averaging across stimuli of the same type
		and considering Subject_type and Stimuli_type as between and within group factors respectively

		>>> analyse(self, standardise_flag=False, average_flag=False, parameter_list={"all"}, between_factor_list=["Subject_type"], within_factor_list=["Stimuli_type"], statistical_test="Mixed_anova", file_creation = True)
		OR
		>>> analyse(self, standardise_flag=True) (as many of the option are present by default)

		For calculating 2-way ANOVA, for "blink_rate" and "avg_blink_duration", without standardisation with averaging across stimuli of the same type
		and considering Subject_type and Gender as the between group factors while NOT creating a new csv file with the results

		>>> analyse(self, average_flag=True, parameter_list={"blink_rate", "avg_blink_duration"}, between_factor_list=["Subject_type", "Gender"], statistical_test="anova", file_creation = False)

		"""

		with open(self.json_file, "r") as json_f:
			json_data = json.load(json_f)

		csvFile = None
		if file_creation:
			directory_path = json_data["Path"] + "/Results"
			if not os.path.isdir(directory_path):
				os.mkdir(directory_path)

			if not os.path.isdir(directory_path + '/Data/'):
				os.mkdir(directory_path + '/Data/')

			if statistical_test != None:
				file_path = directory_path + "/" + statistical_test + ".csv"
				csvFile = open(file_path, 'w')
				writer = csv.writer(csvFile)


		meta_not_to_be_considered = ["pupil_size", "pupil_size_downsample"]

		sacc_flag=0
		ms_flag=0

		for sen in self.sensors:
			for meta in Sensor.meta_cols[sen]:
				if meta in meta_not_to_be_considered:
					continue

				if ('all' not in parameter_list) and (meta not in parameter_list):
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
				column_list.append("stimuli_name")

				data =  pd.DataFrame(columns=column_list)

				#For each subject
				for sub_index, sub in enumerate(self.subjects):
					#For each Question Type
					for stimuli_index, stimuli_type in enumerate(sub.aggregate_meta):

						if meta in ["sacc_duration", "sacc_vel", "sacc_amplitude", "ms_duration", "ms_vel", "ms_amplitude"]:
							summation_array = self.summationArrayCalculation(meta, sub_index, stimuli_index)
						
						value_array = self.meta_matrix_dict[1][meta][sub_index,stimuli_index]

						index_extra = 0

						for value_index, _ in enumerate(value_array):

							if meta in ["sacc_duration", "sacc_vel", "sacc_amplitude", "ms_duration", "ms_vel", "ms_amplitude"]:

								if value_array[value_index] == 0:
									index_extra += 1
									continue

								proper_index = self.return_index(value_index-index_extra, summation_array)
								stimulus_name = self.stimuli[stimuli_type][proper_index]
							else:
								stimulus_name = self.stimuli[stimuli_type][value_index]

							row = []
							row.append(value_array[value_index])

							#Add the between group factors (need to be defined in the json file)
							for param in between_factor_list:

								if param == "Subject_type":
									row.append(sub.subj_type)
									continue

								try:
									row.append(json_data["Subjects"][sub.subj_type][sub.name][param])
								except:
									print("Between subject paramter: ", param, " not defined in the json file")

							for param in within_factor_list:

								if param == "Stimuli_type":
									row.append(stimuli_type)
									continue

								try:
									stimulus_name = self.stimuli[stimuli_type][value_index]
									row.append(json_data["Stimuli"][stimuli_type][stimulus_name][param])
								except:
									print("Within stimuli parameter: ", param, " not defined in the json file")

							row.append(sub.name)
							row.append(stimulus_name)

							if np.isnan(value_array[value_index]):
								print("The data being read for analysis contains null value: ", row)

							#Instantiate into the pandas dataframe
							data.loc[len(data)] = row

				data.to_csv(directory_path + '/Data/' + meta + "_data.csv")

				#print(data)

				#Depending on the parameter, choose the statistical test to be done
				if statistical_test == "Mixed_anova":

					if len(within_factor_list)>1:
						print("Error: Too many within group factors,\nMixed ANOVA can only accept 1 within group factor\n")
					elif len(between_factor_list)>1:
						print("Error: Too many between group factors,\nMixed ANOVA can only accept 1 between group factor\n")

					print(meta, ":\tMixed ANOVA")
					aov = pg.mixed_anova(dv=meta, within=within_factor_list[0], between=between_factor_list[0], subject='subject', data=data)
					pg.print_table(aov)

					if file_creation:

						values_list = ["Mixed Anova: "]
						values_list.append(meta)
						self.fileWriting(writer, csvFile, aov, values_list)

					posthocs = pg.pairwise_ttests(dv=meta, within=within_factor_list[0], between=between_factor_list[0], subject='subject', data=data)
					pg.print_table(posthocs)

					if file_creation:

						values_list = ["Post Hoc Analysis"]
						self.fileWriting(writer, csvFile, posthocs, values_list)

				elif statistical_test == "RM_anova":

					if len(within_factor_list)>2 or len(within_factor_list)<1:
						print("Error: Too many or too few within group factors,\nRepeated Measures ANOVA can only accept 1 or 2 within group factors\n")

					print(meta, ":\tRM ANOVA")
					aov = pg.rm_anova(dv=meta, within= within_factor_list, subject = 'subject', data=data)
					pg.print_table(aov)

					if file_creation:

						values_list = ["Repeated Measures Anova: "]
						values_list.append(meta)
						self.fileWriting(writer, csvFile, aov, values_list)

				elif statistical_test == "anova":

					print(meta, ":\tANOVA")
					length = len(between_factor_list)
					model_equation = meta + " ~ C("

					for factor_index, _ in enumerate(between_factor_list):
						if(factor_index<length-1):
							model_equation = model_equation + between_factor_list[factor_index] + ")*C("
						else:
							model_equation = model_equation + between_factor_list[factor_index] + ")"

					print("Including interaction effect")
					print(model_equation)
					model = ols(model_equation, data).fit()
					res = sm.stats.anova_lm(model, typ= 2)
					print(res)

					if file_creation:

						values_list = ["Anova including interaction effect: "]
						values_list.append(meta)
						self.fileWriting(writer, csvFile, res, values_list)

					print("\nExcluding interaction effect")
					model_equation = model_equation.replace("*", "+")
					print(model_equation)
					model = ols(model_equation, data).fit()
					res = sm.stats.anova_lm(model, typ= 2)
					print(res)

					if file_creation:

						values_list = ["Anova excluding interaction effect: "]
						values_list.append(meta)
						self.fileWriting(writer, csvFile, res, values_list)

				elif statistical_test == "ttest":

					print(meta, ":\tt test")

					if ttest_type==1:
						aov = pg.pairwise_ttests(dv=meta, between=between_factor_list, subject='subject', data=data)
						pg.print_table(aov)
					elif ttest_type==2:
						aov = pg.pairwise_ttests(dv=meta, within=within_factor_list, subject='subject', data=data)
						pg.print_table(aov)
					elif ttest_type==3:
						aov = pg.pairwise_ttests(dv=meta, between=between_factor_list, within=within_factor_list, subject='subject', data=data)
						pg.print_table(aov)
					else:
						print("The value given to ttest_type is not acceptable, it must be either 1 or 2 or 3")


					if file_creation:

						values_list = ["Pairwise ttest: "]
						values_list.append(meta)
						self.fileWriting(writer, csvFile, aov, values_list)

				elif statistical_test == "welch_ttest":

					print(meta, ":\tWelch t test")

					if ttest_type==1:
						normality,aov = self.welch_ttest(dv=meta, factor=between_factor_list[0], subject='subject', data=data)
						pg.print_table(normality)
						pg.print_table(aov)
					elif ttest_type==2:
						normality,aov = self.welch_ttest(dv=meta, factor=within_factor_list[0], subject='subject', data=data)
						pg.print_table(normality)
						pg.print_table(aov)
					else:
						print("The value given to ttest_type for welch test is not acceptable, it must be either 1 or 2")

					if file_creation:

						values_list = ["Welch Pairwise ttest: "]
						values_list.append(meta)
						self.fileWriting(writer, csvFile, normality, values_list)
						self.fileWriting(writer, csvFile, aov, values_list)


		if csvFile != None:
			csvFile.close()


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
			sub_ind = -1
			for ind, subject in enumerate(self.subjects):
				if subject.name == sub:
					sub_ind = ind
					break
			return self.subjects[sub_ind].aggregate_meta

		else:
			sub_ind = -1
			for ind, subject in enumerate(self.subjects):
				if subject.name == sub:
					sub_ind = ind
					break

			stim_cat = ""
			stim_ind = -1
			for cat in self.stimuli:
				if stim in self.stimuli[cat]:
					stim_ind = self.stimuli[cat].index(stim)
					stim_cat = cat
					break

			return self.subjects[sub_ind].stimulus[stim_cat][stim_ind].sensors[sensor].metadata


	def drawAOI(self):
		"""Function that allows speicification of area of interest (AOI) for analysis.

		"""

		with open(self.json_file, "r") as f:
			json_data = json.load(f)

		aoi_left_x = 0
		aoi_left_y = 0
		aoi_right_x = 0
		aoi_right_y = 0

		display_width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
		display_height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]

		cnt = 0
		img = None

		if os.path.isdir(self.path + "/Stimuli/"):
			for f in os.listdir(self.path + "/Stimuli/"):
				if f.split(".")[-1] in ['jpg', 'jpeg', 'png']:
					img = plt.imread(self.path + "/Stimuli/" + f)
					cnt += 1
					break

		if cnt == 0:
			img = np.zeros((display_height, display_width, 3))

		fig, ax = plt.subplots()
		fig.canvas.set_window_title("Draw AOI")
		ax.imshow(img)

		if self.aoi == "p":

			def onselect(verts):
				nonlocal vertices, canvas
				print('\nSelected points:')
				x = []
				y = []
				for i, j in vertices:
					print(round(i, 3), ",", round(j, 3))
					x.append(i)
					y.append(j)

				vertices = verts
				canvas.draw_idle()

			canvas = ax.figure.canvas
			_ = PolygonSelector(ax, onselect, lineprops=dict(color='r', linestyle='-', linewidth=2, alpha=0.5), markerprops=dict(marker='o', markersize=7, mec='r', mfc='k', alpha=0.5))
			vertices = []

			print("1) 'esc' KEY: START A NEW POLYGON")
			print("2) 'shift' KEY: MOVE ALL VERTICES BY DRAGGING ANY EDGE")
			print("3) 'ctrl' KEY: MOVE A SINGLE VERTEX")

			plt.show()
			return vertices

		elif self.aoi == "r":
			def line_select_callback(eclick, erelease):
				nonlocal aoi_left_x, aoi_left_y, aoi_right_x, aoi_right_y
				aoi_left_x, aoi_left_y = round(eclick.xdata, 3), round(eclick.ydata, 3)
				aoi_right_x, aoi_right_y = round(erelease.xdata, 3), round(erelease.ydata, 3)
				print("Coordinates [(start_x, start_y), (end_x, end_y)]: ", "[(%6.2f, %6.2f), (%6.2f, %6.2f)]" % (aoi_left_x, aoi_left_y, aoi_right_x, aoi_right_y))

			RS = RectangleSelector(ax, line_select_callback, drawtype='box', useblit=False, interactive=True)
			RS.to_draw.set_visible(True)

			plt.show()
			return [aoi_left_x, aoi_left_y, aoi_right_x, aoi_right_y]

		elif self.aoi == "e":
			x_dia = 0
			y_dia = 0
			centre = (0,0)
			def onselect(eclick, erelease):
				nonlocal x_dia, y_dia, centre
				x_dia = (erelease.xdata - eclick.xdata)
				y_dia = (erelease.ydata - eclick.ydata)
				centre = [round(eclick.xdata + x_dia/2., 3), round(eclick.ydata + y_dia/2., 3)]
				print("Centre: ", centre)
				print("X Diameter: ", x_dia)
				print("Y Diameter: ", y_dia)
				print()

			ES = EllipseSelector(ax, onselect, drawtype='box', interactive=True, lineprops=dict(color='g', linestyle='-', linewidth=2, alpha=0.5), marker_props=dict(marker='o', markersize=7, mec='g', mfc='k', alpha=0.5))
			plt.show()

			return [centre, x_dia, y_dia]