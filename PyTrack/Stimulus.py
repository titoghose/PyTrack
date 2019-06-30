# -*- coding: utf-8 -*-

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from matplotlib import Path
from matplotlib.patches import Ellipse, Rectangle, Polygon
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy import signal, io, stats, misc
from scipy.ndimage.filters import gaussian_filter
import math

from PyTrack.Sensor import Sensor

class Stimulus:
	"""This is the main class that performs analysis and visualization of data collected during presentation of various stimuli during the experiment.

	If the framework is used in the *Experiment Design* objects of this class are created implicitly and the user need not worry about the internal functioning of the class methods. However, if using the *Stand-alone Design*, the user needs to explicitly create an object of the class and invoke the functions based on what is needed.

	Parameters
	----------
	path : str
		This parameter is the absolute path to the experiment directory containing the json file, stimuli folder with pictures etc. For the *Experiment Design*, this parameter is internally handled. In the *Stand-alone Design*, this parameter needs to be specified while creating the object. All data and plots, if saved, will be stored in folders in this location.
	name : str
		The name of the stimulus. For the *Experiment Design*, this parameter is internally handled. If using in *Stand-alone Design*, this parameter is optional and will default to `id_rather_not`.
	stim_type : str
		The type of stimulus, if there are different classes of stimulus in the experiment. For the *Experiment Design*, this parameter is internally handled. If using in *Stand-alone Design*, this parameter is optional and will default to `doesnt_matter`.
	sensor_names : list(str) | dict
		In the *Experiment Design* `sensor_names` will default to the sensors being used (as mentioned in the json_file). As of now this only supports EyeTracker. If using in *Stand-alone Design*, this parameter must be a dictionary of dictionaries with the details of the sensors and their attributes. The framework as of now just supports Eye Tracking so in the *Stand-alone Design*, `sensor_names` should be in this format (edit the value of the "Sampling_Freq" according to your eye tracker's value):
		{"EyeTracker": {"Sampling_Freq":1000}}
	data : pandas DataFrame
		The data for this stimulus as a Pandas DataFrame. For the *Experiment Design*, this parameter is internally handled. In the *Stand-alone Design* use the `formatBridge.generateCompatibleFormat` module to convert your data into the accepted format and then pass the csv file as a Pandas DataFrame to `data`. This should be the data for just a single stimulus or else the features extracted will not make sense to you. In case you wish to analyse all stimuli for 1 subject, we suggest using the *Experiment Design*.
	start_time : int
		The onset of stimulus. For the *Experiment Design*, this parameter is internally handled and if -1, it implies that data for stimulus is missing. In the *Stand-alone Design*, this parameter is optional (0 by default) and need not be mentioned. However, if supplying an entire dataframe and it is desired to analyse data in a given range, supply the index value to start from. Also specify `end_time` or else -1 is used by default i.e end of DataFrame.
	end_time : int
		The offset of stimulus. For the *Experiment Design*, this parameter is internally handled and if -1, it implies that data for stimulus is missing. In the *Stand-alone Design*, this parameter is optional (-1 by default) and need not be mentioned. However, if supplying an entire dataframe and it is desired to analyse data in a given range, supply the index value to end at. Also specify `start_time` or else 0 is used by default i.e start of DataFrame.
	roi_time : int

	json_file : str
		Desciption of experiment as JSON file. For the *Experiment Design*, this parameter is internally handled. In the *Stand-alone Design* it is not required (leave as ``None``).
	subject_name : str (optional)
		Name of the subject being analysed. For the *Experiment Design*, this parameter is internally handled. In the *Stand-alone Design* it is optional (Defaults to buttersnaps).
	aoi : tuple
		Coordinates of AOI in the following order (start_x, start_y, end_x, end_y). Here, x is the horizontal axis and y is the vertical axis.

	"""


	def __init__(self, path, name="id_rather_not", stim_type="doesnt_matter", sensor_names=["EyeTracker"], data=None, start_time=0, end_time=-1, roi_time=-1, json_file=None, subject_name="buttersnaps", aoi=None):

		path = path.replace("\\", "/")

		self.name = name
		self.path = path
		self.stim_type = stim_type
		self.start_time = start_time
		self.end_time = end_time
		self.response_time = self.end_time - self.start_time
		self.roi_time = roi_time
		self.json_file = json_file
		self.sensors = dict()
		self.subject_name = subject_name

		if not os.path.isdir(self.path + '/Subjects/'):
			os.makedirs(self.path + '/Subjects/')

		if not os.path.isdir(self.path + '/Subjects/' + self.subject_name + '/'):
			os.makedirs(self.path + '/Subjects/' + self.subject_name + '/')

		# Experiment json file exists so stimulus is being created for experiment
		if self.json_file != None:
			self.aoi_coords = aoi
			with open(self.json_file) as json_f:
				json_data = json.load(json_f)
			self.width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
			self.height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]
			if self.start_time == -1:
				self.data = None
			else:
				self.data = self.getData(data, sensor_names)

		# Experiment json file does not exist so stimulus is being created as a stand alone object
		else:
			self.aoi_coords = sensor_names["EyeTracker"]["aoi"]
			self.width = sensor_names["EyeTracker"]["Display_width"]
			self.height = sensor_names["EyeTracker"]["Display_height"]
			self.data = self.getDataStandAlone(data, sensor_names)


	def diff(self, series):
		"""Python implementation of Matlab's 'diff' function.

		Computes the difference between (n+1)th and (n)th elements of array. Returns (a[n+1] - a[n]) for all n.

		Parameters
		----------
		series : list | array (numeric)
			Numeric list, of type ``int`` or ``float``. Must be atleast of length 2.

		Returns
		-------
		list | array (numeric)
			The size of the returned list is n-1 where n is the size of `series` supplied to the `diff`.

		"""
		return series[1:] - series[:-1]


	def smooth(self, x, window_len):
		"""Smoothing function to compute running average.

		Computes the running average for a window size of `window_len`. For the boundary values (`window_len`-1 values at start and end) the window length is reduced to accommodate no padding.

		Parameters
		----------
		x : list | array (numeric)
			Numeric list, of type ``int`` or ``float`` to compute running average for.
		window_len : int
			Size of averaging window. Must be odd and >= 3.

		Returns
		-------
		y : list | array (numeric)
			Smoothed version `x`.

		"""

		# Running average smoothing
		if window_len < 3:
			return x

		# Window length must be odd
		if window_len%2 == 0:
			window_len += 1

		w = np.ones(window_len)
		y = np.convolve(w, x, mode='valid') / len(w)
		y = np.hstack((x[:window_len//2], y, x[len(x)-window_len//2:]))

		for i in range(0, window_len//2):
			y[i] = np.sum(y[0 : i+i]) / ((2*i) + 1)

		for i in range(len(x)-window_len//2, len(x)):
			y[i] = np.sum(y[i - (len(x) - i - 1) : i + (len(x) - i - 1)]) / ((2*(len(x) - i - 1)) + 1)

		return y


	def findBlinks(self, pupil_size, gaze=None, sampling_freq=1000, concat=False, concat_gap_interval=100, interpolate=False):
		"""Finds indices of occurances of blinks and interpolates pupil size and gaze data.

		Function to find blinks and return blink onset, offset indices and interpolated pupil size data.
		Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise," Behav. Res. Methods, vol. 50, no. 1, pp. 107-114, 2018.
		Goto https://osf.io/jyz43/ for R and Matlab implementation.

		Parameters
		----------
		pupil_size : array | list
			Pupil size data for left or right eye
		gaze : dict of list
			Gaze in x and y direction. {"x" : list , "y" : list }
		sampling_freq : float
			Sampling frequency of eye tracking hardware (Defaults to 1000).
		concat : bool
			Concatenate close blinks/missing trials or not. ``False`` by default. See R. Hershman et. al. for more information
		concat_gap_interval : float
			Minimum interval between successive missing samples/blinks to consider for concatenation. If `concat` is ``False`` this parameter does not matter. Default value is 100.
		interpolate : bool
			Interpolate pupil and gaze data durig blinks (Defaults to `False``).

		Returns
		-------
		blinks : dict
			Blink onset and offset indices. {"blink_onset" : list , "blink_offset" : list}
		interp_pupil_size : array | list
			Interpolated pupil size data for left or right eye after fixing blinks. If `interpolate`=``False``, this is the same as `pupil_size` supplied to the function.
		new_gaze : dict
			Interpolated gaze in x and y direction after fixing blinks. If `interpolate`=``False``, this is the same as `gaze` supplied to the function. {"x" : list, "y" : list}

		"""

		blink_onset = []
		blink_offset = []
		blinks = {"blink_onset": blink_onset, "blink_offset": blink_offset}

		sampling_interval = 1000 // sampling_freq

		missing_data = np.array(pupil_size == -1, dtype="float32")
		difference = self.diff(missing_data)

		blink_onset = np.where(difference == 1)[0]
		blink_offset = np.where(difference == -1)[0] + 1

		blink_onset = sorted(blink_onset)
		blink_offset = sorted(blink_offset)

		length_blinks = len(blink_offset) + len(blink_onset)

		if (length_blinks == 0):
			return blinks, pupil_size, gaze

		# Edge Case 2: the data starts with a blink. In this case, blink onset will be defined as the first missing value.
		if ((len(blink_onset) > 0 and blink_onset[0] > 0) or (len(blink_onset) < len(blink_offset))) and pupil_size[0] == -1:
			blink_onset = np.hstack((0, blink_onset))

		# Edge Case 3: the data ends with a blink. In this case, blink offset will be defined as the last missing sample
		if (len(blink_offset) < len(blink_onset)) and pupil_size[-1] == -1:
			blink_offset = np.hstack((blink_offset, len(pupil_size) - 1))

		ms_4_smoothing = 10
		samples2smooth = ms_4_smoothing // sampling_interval
		smooth_pupil_size = np.array(self.smooth(pupil_size, samples2smooth), dtype='float32')

		smooth_pupil_size[np.where(smooth_pupil_size == -1)[0]] = float('nan')
		smooth_pupil_size_diff = self.diff(smooth_pupil_size)

		monotonically_dec = smooth_pupil_size_diff <= 0
		monotonically_inc = smooth_pupil_size_diff >= 0

		for i in range(len(blink_onset)):
			# Edge Case 2: If data starts with blink we do not update it and let starting blink index be 0
			if blink_onset[i] != 0:
				j = blink_onset[i] - 1
				while j > 0 and monotonically_dec[j] == True:
					j -= 1
				blink_onset[i] = j + 1

			# Edge Case 3: If data ends with blink we do not update it and let ending blink index be the last index of the data
			if blink_offset[i] != len(pupil_size) - 1:
				j = blink_offset[i]
				while j < len(monotonically_inc) and monotonically_inc[j] == True:
					j += 1
				blink_offset[i] = j

		if concat:
			c = np.empty((len(blink_onset) + len(blink_offset),))
			c[0::2] = blink_onset
			c[1::2] = blink_offset
			c = list(c)

			i = 1
			while i<len(c)-1:
				if c[i+1] - c[i] <= concat_gap_interval:
					c[i:i+2] = []
				else:
					i += 2

			temp = np.reshape(c, (-1, 2), order='C')
			blink_onset = np.array(temp[:, 0], dtype=int)
			blink_offset = np.array(temp[:, 1], dtype=int)

		blinks["blink_onset"] = blink_onset
		blinks["blink_offset"] = blink_offset

		if interpolate:

			pupil_size_no_blinks = []
			pupil_size_no_blinks_indices = []

			prev = 0

			for i in range(len(blink_onset)):
				if i == 0:
					pupil_size_no_blinks_indices = np.arange(prev, blink_onset[i])
					pupil_size_no_blinks = pupil_size[range(prev, blink_onset[i])]
				else:
					prev = blink_offset[i - 1]
					pupil_size_no_blinks_indices = np.hstack((pupil_size_no_blinks_indices, np.arange(prev, blink_onset[i])))
					pupil_size_no_blinks = np.hstack((pupil_size_no_blinks, pupil_size[range(prev, blink_onset[i])]))

			pupil_size_no_blinks_indices = np.hstack(
				(pupil_size_no_blinks_indices, np.arange(blink_offset[i], len(pupil_size))))
			pupil_size_no_blinks = np.hstack((pupil_size_no_blinks, pupil_size[range(blink_offset[i], len(pupil_size))]))


			interp_pupil_size = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices),
										  pupil_size_no_blinks)

			if gaze != None:
				new_gaze = {"left" : None, "right" : None}
				for eye in ["left", "right"]:
					gaze_x = gaze[eye]["x"]
					gaze_y = gaze[eye]["y"]

					gaze_x_no_blinks = gaze_x[pupil_size_no_blinks_indices]
					gaze_y_no_blinks = gaze_y[pupil_size_no_blinks_indices]

					interp_gaze_x = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices), gaze_x_no_blinks)
					interp_gaze_y = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices), gaze_y_no_blinks)

					new_gaze[eye] = {"x": interp_gaze_x, "y": interp_gaze_y}

			# fig = plt.figure()
			# ax = fig.add_subplot(121)
			# ax2 = fig.add_subplot(122)
			# ax.plot(range(len(pupil_size)), pupil_size, alpha=0.4)
			# ax.plot(range(len(interp_pupil_size)), interp_pupil_size, alpha=0.8, linestyle='--')

			# for i, j in zip(blink_onset, blink_offset):
			# 	ax.axvline(i, color='g', linestyle='--')
			# 	ax.axvline(j, color='g', linestyle='--')
			# 	ax2.plot(gaze["left"]["x"][i:j], gaze["left"]["y"][i:j], color='g', alpha=0.4)


			# rect = Rectangle((self.aoi_coords[0], self.aoi_coords[1]),
			# 						(self.aoi_coords[2] - self.aoi_coords[0]),
			# 						(self.aoi_coords[3] - self.aoi_coords[1]),
			# 						color='r', fill=False, linestyle='--')
			# ax2 = plt.gca()
			# ax2.add_patch(rect)
			# # ax2.plot(gaze["left"]["x"], gaze["left"]["y"], alpha=0.4)
			# ax2.plot(new_gaze["left"]["x"], new_gaze["left"]["y"], alpha=0.8, linestyle='--')

			# plt.show()

		else:
			interp_pupil_size = pupil_size
			new_gaze = gaze

		return blinks, interp_pupil_size, new_gaze


	def findFixations(self):
		"""Function to extract indices of fixation sequences.

		Internal function of class that uses its `data` member variable to compute indices. Does not take any input and can be invoked by an object of the class. Serves as a helper function.

		Returns
		-------
		fixation_indices : dict
			Indices of start and end of fixations. {"start": fixation_onset list, "end": fixation_offset list}

		"""

		fixation_onset = []
		fixation_offset = []

		i = 0

		while i < len(self.data["FixationSeq"]):
			if self.data["FixationSeq"][i] != -1:
				curr = self.data["FixationSeq"][i]
				fixation_onset.append(i)
				while i < len(self.data["FixationSeq"]) and self.data["FixationSeq"][i] != -1 and self.data["FixationSeq"][i] == curr:
					i += 1
				fixation_offset.append(i-1)
			else:
				i += 1

		fixation_indices = {"start": fixation_onset, "end": fixation_offset}

		return fixation_indices


	def findSaccades(self):
		"""Function to extract indices of saccade sequences.

		Saccades are assumed to be interspersed between fixations. Internal function of class that uses its `data` member variable to compute indices. Does not take any input and can be invoked by an object of the class. Serves as a helper function.

		Returns
		-------
		saccade_indices : dict
			Indices of start and end of saccades. {"start": saccade_onset list, "end": saccade_offset list}

		"""

		saccade_onset = []
		saccade_offset = []

		i = 0
		while i < len(self.data["FixationSeq"]):
			if self.data["FixationSeq"][i] == -1:
				saccade_onset.append(i)
				while i < len(self.data["FixationSeq"]) and self.data["FixationSeq"][i] == -1:
					i += 1
				saccade_offset.append(i-1)
			else:
				i += 1

		saccade_indices = {"start": saccade_onset, "end": saccade_offset}

		return saccade_indices


	def position2Velocity(self, gaze, sampling_freq):
		"""Function to calculate velocity for a gaze point based on a 6 sample window.

		Serves as a helper function. See `findMicrosaccades <#Stimulus.Stimulus.findMicrosaccades>`_.

		Parameters
		----------
		gaze : array | list
			Gaze positons in x or y direction.
		sampling_freq : float
			Sampling Frequency of eye tracker.

		Returns
		-------
		velocity : array | list
			Gaze velocities in x or y direction.

		"""

		n = len(gaze)
		velocity = np.zeros(gaze.shape)

		velocity[2:n-2] = (gaze[4:n] + gaze[3:n-1] - gaze[1:n-3] - gaze[0:n-4]) * (sampling_freq/6.0)
		velocity[1] = (gaze[2] - gaze[0]) * (sampling_freq/2.0)
		velocity[n-2] = (gaze[n-1] - gaze[n-3]) * (sampling_freq/2.0)

		return velocity


	def smoothGaze(self, vel, gaze, sampling_freq):
		"""Function to smoothen gaze positions using running average method.

		Serves as a helper function. See `findMicrosaccades <#Stimulus.Stimulus.findMicrosaccades>`_

		Parameters
		----------
		vel : array | list
			Gaze velocities in x or y direction
		gaze : array | list
			Gaze positons in x or y direction
		sampling_freq : float
			Sampling Frequency of eye tracker

		Returns
		-------
		smooth_gaze : array | list
			Smoothened gaze positons in x or y direction

		"""

		smooth_gaze = np.zeros(gaze.shape)

		smooth_gaze[0] = gaze[0]
		vel[1] = vel[1] + smooth_gaze[0]
		smooth_gaze = np.cumsum(vel)

		return smooth_gaze


	def calculateMSThreshold(self, vel, sampling_freq, VFAC=5.0):
		"""Function to calculate velocity threshold value for X and Y directions to classify point as a microsaccade point.

		Serves as a helper function. See `findMicrosaccades <#Stimulus.Stimulus.findMicrosaccades>`_

		Parameters
		---------
		vel : array | list
			Gaze velocity in x or y direction
		sampling_freq : float
			Sampling frequency of the eye tracking device
		VFAC : float
			Scalar constant used to find threshold (Defaults to 5.0). See R. Engbert and K. Mergenthaler, “Microsaccades are triggered by low retinal image slip,” Proc. Natl. Acad. Sci., vol. 103, no. 18, pp. 7192–7197, 2006.

		Returns
		-------
		radius : float
			Threshold radius in x or y direction

		"""

		medx = np.median(vel)
		msdx = np.sqrt(np.median((vel-medx)**2))

		if msdx<1e-10:
			msdx = np.sqrt(np.mean(vel**2) - (np.mean(vel))**2)

		radius = VFAC * msdx
		return radius


	def findBinocularMS(self, msl, msr):
		"""Function to find binocular microsaccades from monocular microsaccades.

		Serves as helper function. See `findMicrosaccades <#Stimulus.Stimulus.findMicrosaccades>`_

		Parameters
		----------
		msl : array | list (num_ms, 9)
			Microsaccade list returned by `findMonocularMS` for the left eye. `num_ms` stands for the number of left eye microsaccades.
		msr : array | list (num_ms, 9)
			Microsaccade list returned by `findMonocularMS` for the right eye. `num_ms` stands for the number of right eye microsaccades.

		Returns
		-------
		ms : dict
			Dictionary of values containing the number of binary microsaccades, number of left eye microsaccades, number of right eye microsaccades, binary microsaccades list, left microsaccades list and right microsaccades list.
			- "NB" : int
			- "NR" : int
			- "NL" : int
			- "bin" : array | list (num_ms, 18)
			- "left" : array | list (num_ms, 9)
			- "right" : array | list (num_ms, 9)

		"""

		numr = len(msl)
		numl = len(msr)

		bin_ms = np.zeros((1, 18))
		monol = np.zeros((1, 9))
		monor = np.zeros((1, 9))

		NB = 0
		NR = 0
		NL = 0

		if (numr * numl) > 0:
			TL = np.max(msl[:, 1])
			TR = np.max(msr[:, 1])
			TB = int(np.max((TL, TR)))
			s = np.zeros(TB+2)

			for left_coords in msl:
				s[int(left_coords[0]) : int(left_coords[1]) + 1] = 1

			for right_coords in msr:
				s[int(right_coords[0]) : int(right_coords[1]) + 1] = 1

			s[0] = 0
			s[TB+1] = 0

			onoff = np.where(self.diff(s) != 0)[0]
			m = np.reshape(onoff, (-1, 2))
			N = m.shape[0]

			for i in range(N):
				left = np.where((msl[:, 0] >= m[i, 0]) & (msl[:, 1] <= m[i, 1]))[0]
				right = np.where((msr[:, 0] >= m[i, 0]) & (msr[:, 1] <= m[i, 1]))[0]

				if (len(right) * len(left)) > 0:
					ampr = np.sqrt((msr[right, 5]**2 + msr[right, 6]**2))
					ampl = np.sqrt((msl[left, 5]**2 + msl[left, 6]**2))

					ir = np.argmax(ampr)
					il = np.argmax(ampl)
					NB += 1
					if NB == 1:
						bin_ms[0][0:9] = msl[left[il], :]
						bin_ms[0][9:18] = msr[right[ir], :]
					else:
						bin_ms = np.vstack((bin_ms, np.hstack((msl[left[il], :], msr[right[ir], :]))))
				else:
					if len(right) == 0:
						NL += 1
						ampl = np.sqrt((msl[left, 5]**2 + msl[left, 6]**2))
						il = np.argmax(ampl)
						if NL == 1:
							monol[0] = msl[left[il], :]
						else:
							monol = np.vstack((monol, msl[left[il], :]))

					if len(left) == 0:
						NR += 1
						ampr = np.sqrt((msr[right, 5]**2 + msr[right, 6]**2))
						ir = np.argmax(ampr)
						if NR == 1:
							monor[0] = msr[right[ir], :]
						else:
							monor = np.vstack((monor, msr[right[ir], :]))

		else:
			if numr == 0:
				bin_ms = None
				monor = None
				monol = msl
			if numl == 0:
				bin_ms = None
				monor = msr
				monol = None

		ms = {"NB" : NB, "NR" : NR, "NL" : NL, "bin" : bin_ms, "left" : monol, "right" : monor}

		return ms


	def findMonocularMS(self, gaze, vel, sampling_freq=1000):
		"""Function to find binocular microsaccades from monocular microsaccades.

		Serves as helper function. See `findMicrosaccades <#Stimulus.Stimulus.findMicrosaccades>`_

		Parameters
		----------
		gaze : array | list
			Gaze positons in x or y direction
		vel : array | list
			Gaze velocities in x or y direction
		sampling_freq : float
			Sampling Frequency of eye tracker (Defaults to 1000)

		Returns
		-------
		MS : array (num_ms, 9)
			Array of 9 microsaccade Parameters. These Parameters correspond to the following array indices
			0. starting index
			1. ending index
			2. peak velocity
			3. microsaccade gaze vector (x direction)
			4. microsaccade gaze vector (y direction)
			5. amplitude (x direction)
			6. amplitude (y direction)
			7. threshold radius (x direction)
			8. threshold radius (y direction)
			num_ms = `ms_count`
		ms_count : int
			Number of microsaccades
		ms_duration : list(int)
			List of duration of each microsaccade. Contains as many values as `ms_count`

		"""

		MINDUR = int((sampling_freq / 1000) * 6)
		gaze_x = gaze["x"]
		gaze_y = gaze["y"]

		vel_x = vel["x"]
		vel_y = vel["y"]

			# for i in range(len(fixation_indices["start"])):
		# 	print(fixation_indices["start"][i], fixation_indices["end"][i])

		radius_x = self.calculateMSThreshold(vel_x, sampling_freq)
		radius_y = self.calculateMSThreshold(vel_y, sampling_freq)

		temp = (vel_x/radius_x)**2 + (vel_y/radius_y)**2
		ms_indices = np.where(temp > 1)[0]

		# for ind, msi in enumerate(ms_indices):
		# 	print(ind, msi)

		N = len(ms_indices)
		num_ms = 0
		MS = np.zeros((1, 9))
		duration = 1
		a = 0
		k = 0

		# Loop over saccade candidates
		while k<N-1:
			if (ms_indices[k+1] - ms_indices[k]) == 1:
				duration += 1
			else:
				# Minimum duration criterion (exception: last saccade)
				if duration >= MINDUR:
					num_ms += 1
					b = k
					if num_ms == 1:
						MS[0][0] = ms_indices[a]
						MS[0][1] = ms_indices[b]
					else:
						new_ms = np.array([ms_indices[a], ms_indices[b], 0, 0, 0, 0, 0, 0, 0])
						MS = np.vstack((MS, new_ms))

				a = k+1
				duration = 1

			k += 1

		# Check minimum duration for last microsaccade
		if duration >= MINDUR:
			num_ms += 1
			b = k
			if num_ms == 1:
				MS[0][0] = ms_indices[a]
				MS[0][1] = ms_indices[b]
			else:
				new_ms = np.array([ms_indices[a], ms_indices[b], 0, 0, 0, 0, 0, 0, 0])
				MS = np.vstack((MS, new_ms))

		if num_ms>0:
			# Compute peak velocity, horiztonal and vertical components
			for s in range(num_ms):

				# Onset and offset for saccades
				a = int(MS[s][0])
				b = int(MS[s][1])
				idx = range(a, b)

				# Saccade peak velocity (vpeak)
				vpeak = max(np.sqrt(vel_x[idx]**2 + vel_y[idx]**2))
				MS[s][2] = vpeak

				# Saccade vector (dx,dy)
				dx = gaze_x[b] - gaze_x[a]
				dy = gaze_y[b] - gaze_y[a]
				MS[s][3] = dx
				MS[s][4] = dy

				# Saccade amplitude (dX,dY)
				minx = min(gaze_x[idx])
				maxx = max(gaze_x[idx])
				miny = min(gaze_y[idx])
				maxy = max(gaze_y[idx])
				ix1 = np.argmin(gaze_x[idx])
				ix2 = np.argmax(gaze_x[idx])
				iy1 = np.argmin(gaze_y[idx])
				iy2 = np.argmax(gaze_y[idx])
				dX = np.sign(ix2 - ix1) * (maxx - minx)
				dY = np.sign(iy2 - iy1) * (maxy - miny)
				MS[s][5] = dX
				MS[s][6] = dY

				MS[s][7] = radius_x
				MS[s][8] = radius_y

			ms_count = num_ms

		ms_count = num_ms
		ms_duration = []

		for ms in MS:
			ms_duration.append(ms[1] - ms[0])

		if num_ms == 0:
			MS = []

		return np.array(MS), ms_count, ms_duration


	def findMicrosaccades(self, sampling_freq=1000, plot_ms=False):
		"""Function to detect microsaccades within fixations.

		Adapted from R. Engbert and K. Mergenthaler, “Microsaccades are triggered by low retinal image slip,” Proc. Natl. Acad. Sci., vol. 103, no. 18, pp. 7192–7197, 2006.

		Parameters
		----------
		sampling_freq : float
			Sampling Frequency of eye tracker (Defaults to 1000)
		plot_ms : bool
			Wether to plot microsaccade plots and main sequence or not (Defaults to ``False``). If ``True``, the figures will be plot and saved in the folder Subjects in the experiment folder.

		Returns
		-------
		all_bin_MS : return value of `findBinocularMS`
			All the binocular microsaccades found for the given stimuli.
		ms_count : int
			Total count of all binocular and monocular microsaccades.
		ms_duration : list(sloat)
			List of durations of all microsaccades.
		temp_vel : list(float)
			List of peak velocities of all microsaccades.
		temp_amp : list(float)
			List of amplitudes of all microsaccades.

		"""
		fixation_indices = self.findFixations()
		all_bin_MS = []

		if plot_ms:
			fig2 = plt.figure()
			fig2.add_subplot(111)

		for fix_ind in range(len(fixation_indices["start"])):

			all_MS = {"left" : None, "right" : None}
			ms_count = {"left" : None, "right" : None}
			ms_duration = {"left" : None, "right" : None}
			smooth_gaze = {"left" : None, "right" : None}
			vel = {"left" : None, "right" : None}


			for i in ["left", "right"]:

				curr_gaze = {"x" : self.data["Gaze"][i]["x"][fixation_indices["start"][fix_ind] : fixation_indices["end"][fix_ind] + 1],
							"y" : self.data["Gaze"][i]["y"][fixation_indices["start"][fix_ind] : fixation_indices["end"][fix_ind] + 1]}

				vel_x = self.position2Velocity(curr_gaze["x"], sampling_freq)
				vel_y = self.position2Velocity(curr_gaze["y"], sampling_freq)
				temp_vel = {"x" : vel_x, "y" : vel_y}
				vel[i] = temp_vel

				smooth_gaze_x = self.smoothGaze(self.position2Velocity(curr_gaze["x"], sampling_freq=1), curr_gaze["x"], sampling_freq)
				smooth_gaze_y = self.smoothGaze(self.position2Velocity(curr_gaze["y"], sampling_freq=1), curr_gaze["y"], sampling_freq)
				temp_smooth_gaze = {"x" : smooth_gaze_x, "y" : smooth_gaze_y}
				smooth_gaze[i] = temp_smooth_gaze

				all_MS[i], ms_count[i], ms_duration[i] = self.findMonocularMS(curr_gaze, vel[i], sampling_freq)

			MS = self.findBinocularMS(all_MS["left"], all_MS["right"])
			all_bin_MS.append(MS)


			if plot_ms and MS["NB"] != 0:

				# Plot gaze and velocity with thresholds
				fig = plt.figure()
				a1 = fig.add_subplot(1, 2, 1)
				a2 = fig.add_subplot(1, 2, 2)

				plt.subplots_adjust(wspace=0.5)

				# Plotting positions
				a1.plot(smooth_gaze["left"]["x"][1:], smooth_gaze["left"]["y"][1:])
				a1.set_xlabel("x")
				a1.set_ylabel("y")
				a1.set_title("Gaze Plot	")
				for i in range(MS["NB"]):
					a1.plot(smooth_gaze["left"]["x"][int(MS["bin"][i][0]) : int(MS["bin"][i][1]) + 1], smooth_gaze["left"]["y"][int(MS["bin"][i][0]) : int(MS["bin"][i][1]) + 1], color='r')
				# a1.set_xlim([-0.35, 0.25])
				# a1.set_ylim([-0.2, 1.25])

				e = Ellipse((0, 0), 2*MS["bin"][0][7], 2*MS["bin"][0][8], linestyle='--', color='g', fill=False)
				a2.add_patch(e)

				# Plotting velocities
				a2.plot(vel["left"]["x"], vel["left"]["y"], alpha=0.5)
				a2.set_xlabel("vel-x")
				a2.set_ylabel("vel-y")
				a2.set_title("Gaze Velocity Plot")
				for i in range(MS["NB"]):
					a2.plot(vel["left"]["x"][int(MS["bin"][i][0]) : int(MS["bin"][i][1]) + 1], vel["left"]["y"][int(MS["bin"][i][0]) : int(MS["bin"][i][1]) + 1], color='r')
				# a2.set_xlim([-25, 40])
				# a2.set_ylim([-65, 70])

				if not os.path.isdir(self.path + "/Subjects/" + self.subject_name + "/ms_gaze_vel/"):
					os.makedirs(self.path + "/Subjects/" + self.subject_name + "/ms_gaze_vel/")

				fig.savefig(self.path + "/Subjects/" + self.subject_name + "/ms_gaze_vel/" + self.name + "_" + str(fix_ind) + ".png", dpi=200)
				plt.close(fig)

				ax = fig2.axes[0]
				ax.set_xlabel("Amplitude (deg)")
				ax.set_ylabel("Peak Velocity (deg/s)")
				for i in range(MS["NB"]):
					peak_vel = (MS["bin"][i][2] + MS["bin"][i][11])/2
					amp = (np.sqrt(MS["bin"][i][5]**2 + MS["bin"][i][6]**2) + np.sqrt(MS["bin"][i][13]**2 + MS["bin"][i][14]**2))/2
					ax.scatter(amp, peak_vel, marker='o', facecolors='none', edgecolors='r')

		if plot_ms:
			fig2.savefig(self.path + "/Subjects/" + self.subject_name + "/ms_main_seq" + self.name + ".png", dpi=200)
			plt.close(fig2)

		ms_count = 0
		ms_duration = np.zeros(1, dtype='float32')
		temp_vel = np.zeros(1, dtype='float32')
		temp_amp = np.zeros(1, dtype='float32')
		for ms in all_bin_MS:
			# Net microsaccade count i.e binary + left +right
			if ms["NB"] != 0:
				for m in ms["bin"]:
					if len(np.where(self.data["GazeAOI"][int(m[0]) : int(m[1])] == 1)[0]) > int(0.9 * (m[1] - m[0])):
						ms_count += 1
						# Appending peak velocity for binary microsaccade
						vel_val = (m[2] + m[11]) / 2.
						temp_vel = np.hstack((temp_vel, vel_val))

						# Appending amplitude for binary microsaccade
						amp_val = (np.sqrt(m[5]**2 + m[6]**2) + np.sqrt(m[13]**2 + m[14]**2)) / 2.
						temp_amp = np.hstack((temp_amp, amp_val))

						# Appending durations for binary microsaccade
						dur_val = ((m[1] - m[0]) + (m[10] - m[9])) / 2.
						ms_duration = np.hstack((ms_duration, dur_val))

			# if ms["NL"] != 0:
			# 	for m in ms["left"]:
			# 		if len(np.where(self.data["GazeAOI"][int(m[0]) : int(m[1])] == 1)[0]) > int(0.9 * (m[1] - m[0])):
			# 			ms_count += 1
			# 			# Appending peak velocity for left eye microsaccade
			# 			temp_vel = np.hstack((temp_vel, m[2]))

			# 			# Appending amplitude for left eye microsaccade
			# 			temp_amp = np.hstack((temp_amp, np.sqrt(m[5]**2 + m[6]**2)))

			# 			# Appending durations for left eye microsaccade
			# 			dur_val = m[1] - m[0]
			# 			ms_duration = np.hstack((ms_duration, dur_val))

			# if ms["NR"] != 0:
			# 	for m in ms["right"]:
			# 		if len(np.where(self.data["GazeAOI"][int(m[0]) : int(m[1])] == 1)[0]) > int(0.9 * (m[1] - m[0])):
			# 			ms_count += 1
			# 			# Appending peak velocity for left eye microsaccade
			# 			temp_vel = np.hstack((temp_vel, m[2]))

			# 			# Appending amplitude for left eye microsaccade
			# 			temp_amp = np.hstack((temp_amp, np.sqrt(m[5]**2 + m[6]**2)))

			# 			# Appending durations for left eye microsaccade
			# 			dur_val = m[1] - m[0]
			# 			ms_duration = np.hstack((ms_duration, dur_val))

		if ms_count == 0:
			ms_duration = [0, 0]
			temp_vel = [0, 0]
			temp_amp = [0, 0]

		return all_bin_MS, ms_count, ms_duration[1:], temp_vel[1:], temp_amp[1:]


	def findSaccadeParams(self, sampling_freq=1000):
		"""Function to find saccade parameters like peak velocity, amplitude, count and duration.

		Internal function of class that uses its `data` member variable. Serves as a helper function. See `findEyeMetaData <#Stimulus.Stimulus.findEyeMetaData>`_

		Parameters
		----------
		sampling_freq : float
			Sampling Frequency of eye tracker (Defaults to 1000)

		Returns
		-------
		tuple
			Tuple consisting of (saccade_count, saccade_duration, saccade_peak_vel, saccade_amplitude).

		"""

		saccade_indices = self.findSaccades()
		saccade_onset = saccade_indices["start"]
		saccade_offset = saccade_indices["end"]

		saccade_count = 0
		saccade_duration = []

		saccade_peak_vel = []
		saccade_amplitude = []

		for start, end in zip(saccade_onset, saccade_offset):

			if (end-start) < 6:
				continue

			if len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)):

				saccade_count += 1

				saccade_duration.append(end-start)

				vel_x = self.position2Velocity(self.data["InterpGaze"]["left"]["x"][start:end], sampling_freq)
				vel_y = self.position2Velocity(self.data["InterpGaze"]["left"]["y"][start:end], sampling_freq)

				# Saccade peak velocity (vpeak)
				vpeak = max(np.sqrt(vel_x**2 + vel_y**2))
				saccade_peak_vel.append(vpeak)

				# Saccade amplitude (dX,dY)
				minx = min(self.data["InterpGaze"]["left"]["x"][start:end])
				maxx = max(self.data["InterpGaze"]["left"]["x"][start:end])
				miny = min(self.data["InterpGaze"]["left"]["y"][start:end])
				maxy = max(self.data["InterpGaze"]["left"]["y"][start:end])
				ix1 = np.argmin(self.data["InterpGaze"]["left"]["x"][start:end])
				ix2 = np.argmax(self.data["InterpGaze"]["left"]["x"][start:end])
				iy1 = np.argmin(self.data["InterpGaze"]["left"]["y"][start:end])
				iy2 = np.argmax(self.data["InterpGaze"]["left"]["y"][start:end])
				dX = np.sign(ix2 - ix1) * (maxx - minx)
				dY = np.sign(iy2 - iy1) * (maxy - miny)
				saccade_amplitude.append(np.sqrt(dX**2 + dY**2))

		if saccade_count == 0:
			saccade_duration = [0]
			saccade_peak_vel = [0]
			saccade_amplitude = [0]

		return (saccade_count, saccade_duration, saccade_peak_vel, saccade_amplitude)


	def findResponseTime(self, sampling_freq=1000):
		"""Function to find the response time in milliseconds based on the sampling frequency of the eye tracker.

		Internal function of class that uses its `data` member variable. Serves as a helper function. See `findEyeMetaData <#Stimulus.Stimulus.findEyeMetaData>`_

		Parameters
		----------
		sampling_freq : float
			Sampling Frequency of eye tracker (Defaults to 1000)

		Returns
		-------
		float
			Response time in milliseconds

		"""
		return len(self.data["ETRows"] * (1000/sampling_freq))


	def findFixationParams(self):
		"""Function to find fixation parameters like count, max duration and average duration.

		Internal function of class that uses its `data` member variable. Does not take any input and can be invoked by an object of the class. Serves as a helper function. See `findEyeMetaData <#Stimulus.Stimulus.findEyeMetaData>`_

		Returns
		-------
		tuple
			Tuple consisting of (fixation_count, max_fixation_duration, avg_fixation_duration)

		"""

		inside_aoi = [0, 0, 0]

		fix_num, fix_ind, fix_cnt = np.unique(self.data["FixationSeq"], return_index=True, return_counts=True)
		fixation_count = len(fix_num) - 1

		if fixation_count != 0:
			temp1 = []
			fix_ind_end = np.array(fix_ind[1:]) + np.array(fix_cnt[1:])
			for start, end in zip(fix_ind[1:], fix_ind_end):
				if len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)):
					inside_aoi[0] += 1
					temp1.append(end - start)

			if len(temp1) != 0:
				inside_aoi[1] = np.max(temp1)
				inside_aoi[2] = np.mean(temp1)

		return tuple(inside_aoi)


	def findPupilParams(self):
		"""Function to find pupil parameters like size, peak size, time to peak size, area under curve, slope, mean size, downsampled pupil size

		Internal function of class that uses its `data` member variable. Does not take any input and can be invoked by an object of the class. Serves as a helper function. See `findEyeMetaData <#Stimulus.Stimulus.findEyeMetaData>`_

		Returns
		-------
		tuple
			Tuple consisting of (pupil_size, peak_pupil, time_to_peak, pupil_AUC, pupil_slope, pupil_mean, pupil_size_downsample)

		"""

		pupil_size = self.data["InterpPupilSize"] - self.data["InterpPupilSize"][0]
		peak_pupil = max(pupil_size)
		time_to_peak = np.argmax(pupil_size)

		# Finding Area Under Curve (AUC)
		index = np.argmin(pupil_size)
		length = len(pupil_size)
		pupil_AUC = 0
		while index < length:
			if pupil_size[index] < 0:
				pupil_AUC += abs(pupil_size[index])
			index += 1

		# Finding slope of regression line fit on pupil_size data
		x = np.array([i for i in range(1500)])
		y = pupil_size[0:1500]
		pupil_slope, _, _, _, _ = stats.linregress(x[:len(y)],y)

		# Finding mean of pupil_size data
		pupil_mean = np.mean(pupil_size)

		# Finding decimated pupil_size value (at 60Hz)
		frequency = 60
		downsample_stride = int(1000/frequency)
		index = 0
		pupil_size_downsample = []
		while index < length:
			pupil_size_downsample.append(pupil_size[index])
			index += downsample_stride

		return (pupil_size, peak_pupil, time_to_peak, pupil_AUC, pupil_slope, pupil_mean, pupil_size_downsample)


	def findBlinkParams(self):
		"""Function to find blink parameters like count, duration and average duration

		Internal function of class that uses its `data` member variable. Does not take any input and can be invoked by an object of the class. Serves as a helper function. See `findEyeMetaData <#Stimulus.Stimulus.findEyeMetaData>`_

		Returns
		-------
		list
			Tuple consisting of (blink_cnt, peak_blink_duration, avg_blink_duration)

		"""

		inside_aoi = [0, 0, 0]

		blink_cnt = len(self.data["BlinksLeft"]["blink_onset"])
		if blink_cnt != 0:
			temp1 = []
			for start, end in zip(self.data["BlinksLeft"]["blink_onset"], self.data["BlinksLeft"]["blink_offset"]):
				if len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)):
					inside_aoi[0] += 1
					temp1.append(end - start)

			if len(temp1) != 0:
				inside_aoi[1] = np.max(temp1)
				inside_aoi[2] = np.mean(temp1)

		return tuple(inside_aoi)


	def gazePlot(self, save_fig=False, show_fig=True, save_data=False):
		"""Function to plot eye gaze with numbered fixations.

		Internal function of class that uses its `data` member variable. Can be invoked by an object of the class.

		Parameters
		----------
		save_fig : bool
			Save the gaze plot figure or not (Defaults to ``False``). If ``True``, will be saved in the Subjects folder of the experiment folder
		show_fig : bool
			Display the gaze plot figure or not (Defaults to ``True``).
		save_data : bool
			Save the data used for plotting as a csv file.

		"""

		if self.data == None:
			return

		fig = plt.figure()
		fig.canvas.set_window_title("Gaze Plot: " + self.name)

		try:
			img = plt.imread(self.path + "/Stimuli/" + self.name + ".jpg")
		except:
			try:
				img = plt.imread(self.path + "/Stimuli/" + self.name + ".jpeg")
			except:
				img = np.zeros((self.height, self.width))

		ax = plt.gca()
		ax.imshow(img)

		# Rectangle AOI
		if len(self.aoi_coords) == 4 and (isinstance(self.aoi_coords[0], float) or isinstance(self.aoi_coords[0], int)):
			rect = Rectangle((self.aoi_coords[0], self.aoi_coords[1]),
									(self.aoi_coords[2] - self.aoi_coords[0]),
									(self.aoi_coords[3] - self.aoi_coords[1]),
									color='r', fill=False, linestyle='--')
			ax.add_patch(rect)
			print(rect.get_verts())

		# Circle AOI
		elif len(self.aoi_coords) == 3:
			ellipse = Ellipse((self.aoi_coords[0][0], self.aoi_coords[0][1]),
								self.aoi_coords[1],
								self.aoi_coords[2],
								color='r', fill=False, linestyle='--')
			ax.add_patch(ellipse)
			print(ellipse.get_verts())

		# Polygon AOI
		else:
			xy = np.asarray(self.aoi_coords)
			poly = Polygon(xy, color='r', fill=False, linestyle='--')
			ax.add_patch(poly)
			print(poly.get_verts())


		fixation_dict = self.findFixations()
		fixation_indices = np.vstack((fixation_dict["start"], fixation_dict["end"]))
		fixation_indices = np.reshape(fixation_indices, (fixation_indices.shape[0] * fixation_indices.shape[1]), order='F')

		gaze_x = np.array(np.split(self.data["InterpGaze"]["left"]["x"], fixation_indices))
		gaze_y = np.array(np.split(self.data["InterpGaze"]["left"]["y"], fixation_indices))

		fixation_mask = np.arange(start=1, stop=len(gaze_x), step=2)
		saccade_mask = np.arange(start=0, stop=len(gaze_x), step=2)

		fixation_gaze_x = gaze_x[fixation_mask]
		saccade_gaze_x = gaze_x[saccade_mask]

		fixation_gaze_y = gaze_y[fixation_mask]
		saccade_gaze_y = gaze_y[saccade_mask]

		ax.plot(self.data["InterpGaze"]["left"]["x"], self.data["InterpGaze"]["left"]["y"], 'r-')

		i = 0
		exp_data = {"ind":[], "x":[], "y":[], "gaze_x":self.data["InterpGaze"]["left"]["x"], "gaze_y":self.data["InterpGaze"]["left"]["y"]}
		for x, y in zip(fixation_gaze_x, fixation_gaze_y):
			ax.plot(np.mean(x), np.mean(y), 'go', markersize=15, alpha=0.7)
			exp_data["ind"].append(i)
			exp_data["x"].append(np.mean(x))
			exp_data["y"].append(np.mean(y))
			ax.text(np.mean(x), np.mean(y), str(i), fontsize=10, color='w')
			i += 1


		ax.set_xlim(0, int(self.width))
		ax.set_ylim(int(self.height), 0)

		if show_fig:
			plt.show()

		if save_fig:
			fig.savefig(self.path + "/Subjects/" + self.subject_name + "/gaze_plot_" + self.name + ".png", dpi=300)

		if save_data:
			max_len = max([len(exp_data[x]) for x in exp_data])
			for key in exp_data:
				exp_data[key] = np.pad(exp_data[key], (0, max_len - len(exp_data[key])), 'constant', constant_values=float('nan'))
			pd.DataFrame().from_dict(exp_data).to_csv(self.path + "/Subjects/" + self.subject_name + "/gaze_plot_data_" + self.name + ".csv")

		plt.close(fig)


	def gazeHeatMap(self, save_fig=False, show_fig=True, save_data=False):
		"""Function to plot heat map of gaze.

		Internal function of class that uses its `data` member variable. Can be invoked by an object of the class.

		Parameters
		----------
		save_fig : bool
			Save the heat map figure or not (Defaults to ``False``). If ``True``, will be saved in the Subjects folder of the experiment folder
		show_fig : bool
			Display the heat map figure or not (Defaults to ``True``).
		save_data : bool
			Save the data used for plotting as a csv file.

		"""

		if self.data == None:
			return

		fig = plt.figure()
		fig.canvas.set_window_title("Gaze Heat Map: " + self.name)

		ax = plt.gca()

		x = self.data["InterpGaze"]["left"]["x"]
		y = self.data["InterpGaze"]["left"]["y"]

		# In order to get more intense values for the heatmap (ratio of points is unaffected)
		x = np.repeat(x, 10)
		y = np.repeat(y, 10)

		try:
			img = plt.imread(self.path + "/Stimuli/" + self.name + ".jpg")
		except:
			try:
				img = plt.imread(self.path + "/Stimuli/" + self.name + ".jpeg")
			except:
				img = np.zeros((self.height, self.width))

		downsample_fraction = 0.25
		col_shape = img.shape[1]
		row_shape = img.shape[0]

		hist, _, _ = np.histogram2d(y, x, bins=[int(col_shape*downsample_fraction), int(row_shape*downsample_fraction)], range=[[0, int(col_shape)], [0, int(row_shape)]])
		hist = gaussian_filter(hist, sigma=12)
		hist = hist.repeat(int(1/downsample_fraction), axis=0).repeat(int(1/downsample_fraction), axis=1)

		cmap = pl.cm.jet
		my_cmap = cmap(np.arange(cmap.N))
		my_cmap[:cmap.N//4, -1] = 0
		my_cmap[cmap.N//4:, -1] = np.linspace(0.2, 0.4, cmap.N - cmap.N//4)
		my_cmap = ListedColormap(my_cmap)

		ax.imshow(img)

		# Rectangle AOI
		if len(self.aoi_coords) == 4 and (isinstance(self.aoi_coords[0], float) or isinstance(self.aoi_coords[0], int)):
			rect = Rectangle((self.aoi_coords[0], self.aoi_coords[1]),
									(self.aoi_coords[2] - self.aoi_coords[0]),
									(self.aoi_coords[3] - self.aoi_coords[1]),
									color='r', fill=False, linestyle='--')
			ax.add_patch(rect)

		# Circle AOI
		elif len(self.aoi_coords) == 3:
			ellipse = Ellipse((self.aoi_coords[0][0], self.aoi_coords[0][1]),
								self.aoi_coords[1],
								self.aoi_coords[2],
								color='r', fill=False, linestyle='--')
			ax.add_patch(ellipse)

		# Polygon AOI
		else:
			xy = np.asarray(self.aoi_coords)
			poly = Polygon(xy, color='r', fill=False, linestyle='--')
			ax.add_patch(poly)


		ax.contourf(np.arange(0, int(row_shape), 1), np.arange(0, int(col_shape), 1), hist, cmap=my_cmap)
		ax.set_xlim(0, int(col_shape))
		ax.set_ylim(int(row_shape), 0)

		if show_fig:
			plt.show()

		if save_fig:
			fig.savefig(self.path + "/Subjects/" + self.subject_name + "/gaze_heatmap_" + self.name + ".png", dpi=300)

		plt.close(fig)


	def visualize(self, show=True, save_data=False):
		"""Function to create dynamic plot of gaze and pupil size.

		Internal function of class that uses its `data` member variable. Does not take any input and can be invoked by an object of the class.

		Paramaters
		----------
		show : bool
			Open figure after plotting the data or not.
		save_data : bool
			Save the data used for plotting as a csv file.

		"""
		if self.data == None:
			return

		total_range = None

		# Initialising Plots
		fig = plt.figure()
		fig.canvas.set_window_title(self.name)
		ax = fig.add_subplot(2, 1, 1)
		ax2 = fig.add_subplot(2, 1, 2)

		try:
			img = plt.imread(self.path + "/Stimuli/" + self.name + ".jpg")
		except:
			try:
				img = plt.imread(self.path + "/Stimuli/" + self.name + ".jpeg")
			except:
				img = np.zeros((self.height, self.width))

		ax.imshow(img)
		# Rectangle AOI
		if len(self.aoi_coords) == 4 and (isinstance(self.aoi_coords[0], float) or isinstance(self.aoi_coords[0], int)):
			rect = Rectangle((self.aoi_coords[0], self.aoi_coords[1]),
									(self.aoi_coords[2] - self.aoi_coords[0]),
									(self.aoi_coords[3] - self.aoi_coords[1]),
									color='r', fill=False, linestyle='--')
			ax.add_patch(rect)

		# Circle AOI
		elif len(self.aoi_coords) == 3:
			ellipse = Ellipse((self.aoi_coords[0][0], self.aoi_coords[0][1]),
								self.aoi_coords[1],
								self.aoi_coords[2],
								color='r', fill=False, linestyle='--')
			ax.add_patch(ellipse)

		# Polygon AOI
		else:
			xy = np.asarray(self.aoi_coords)
			poly = Polygon(xy, color='r', fill=False, linestyle='--')
			ax.add_patch(poly)

		if self.data["InterpGaze"] != None:
			total_range = range(len(self.data["ETRows"]))
			# Plot for eye gaze
			line, = ax.plot(self.data["InterpGaze"]["left"]["x"][:1], self.data["InterpGaze"]["left"]["y"][:1], 'r-', alpha=1)
			circle, = ax.plot(self.data["InterpGaze"]["left"]["x"][1], self.data["InterpGaze"]["left"]["y"][1], 'go', markersize=10, alpha=0.7)
			ax.set_title("Gaze")

			# Plot for pupil size
			line3, = ax2.plot(total_range[:1], self.data["InterpPupilSize"][:1])
			ax2.set_xlim([0, len(total_range)])
			ax2.set_ylim([-2, 11])
			ax2.set_title("Pupil Size vs. Time")
			ax2.set_xlabel("Time (ms)")
			ax2.set_ylabel("Pupil Size")

			for i in range(len(self.data["BlinksLeft"]["blink_onset"])):
				plt.axvline(x=self.data["BlinksLeft"]["blink_onset"][i], linestyle="--", color="r", alpha=0.4)
				plt.axvline(x=self.data["BlinksLeft"]["blink_offset"][i], linestyle="--", color="g", alpha=0.6)

				plt.axvline(x=self.data["BlinksLeft"]["blink_onset"][i], linestyle="--", color="r", alpha=0.4)
				plt.axvline(x=self.data["BlinksLeft"]["blink_offset"][i], linestyle="--", color="g", alpha=0.6)

		axamp = plt.axes([0.25, .03, 0.50, 0.02])
		samp = Slider(axamp, 'Time', 1, total_range[-1], valinit=0, valstep=1)

		is_manual = False

		def update_slider(val):
			"""
			"""
			nonlocal is_manual
			is_manual = True
			val = int(val)
			update(val)

		def update(i):
			"""
			"""
			i = int(i)

			if self.data["InterpGaze"] != None:
				line.set_xdata(self.data["InterpGaze"]["left"]["x"][:i])
				line.set_ydata(self.data["InterpGaze"]["left"]["y"][:i])

				circle.set_xdata(self.data["InterpGaze"]["left"]["x"][i])
				circle.set_ydata(self.data["InterpGaze"]["left"]["y"][i])

				line3.set_xdata(total_range[:i])
				line3.set_ydata(self.data["InterpPupilSize"][:i])
				ax2.set_ylim([min(self.data["InterpPupilSize"][:i]) - 5, max(self.data["InterpPupilSize"][:i]) + 5])

			fig.canvas.draw_idle()

		def update_plot(i):
			"""
			"""
			nonlocal is_manual
			if is_manual:
				return [line, circle, line3]

			i = int(samp.val + 1) % total_range[-1]
			samp.set_val(i)
			is_manual = False # the above line called update_slider, so we need to reset this
			fig.canvas.draw_idle()

			return [line, circle, line3]


		def on_click(event):
			"""
			"""
			nonlocal is_manual
			# Check where the click happened
			(xm,ym),(xM,yM) = samp.label.clipbox.get_points()
			if (xm < event.x < xM and ym < event.y < yM):
				# Event happened within the slider or checkbox, ignore since it is handled in update_slider
				return
			else:
				# user clicked somewhere else on canvas = unpause
				is_manual=False

		# call update function on slider value change
		samp.on_changed(update_slider)
		fig.canvas.mpl_connect('button_press_event', on_click)

		ani = animation.FuncAnimation(fig, update_plot, interval=1)

		if show:
			plt.show()


	def numberRevisits(self):
		"""Calculates the number of times the eye revisits within the region of interest, each instance should atleast be 4 milliseconds long

		Returns
		-------
		num_readings: int
			Number of times the subject revisits the Area of Interest (1 revisit is consecutive fixations within AOI)

		"""

		num_readings = 0
		flag = 0

		fix_num, fix_ind, fix_cnt = np.unique(self.data["FixationSeq"], return_index=True, return_counts=True)
		fixation_count = len(fix_num) - 1

		if fixation_count != 0:
			temp1 = []
			fix_ind_end = np.array(fix_ind[1:]) + np.array(fix_cnt[1:])
			for start, end in zip(fix_ind[1:], fix_ind_end):

				if len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)) and flag==0:
					num_readings += 1
					flag = 1
				elif len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) < int(0.9 * (end-start)):
					flag = 0

		return num_readings


	def passDurationCalculation(self):
		"""Calculates the amount of time spent during the first and second revisit in the region of interest

		Returns
		-------
		first_pass_duration: int
			duration spent on first visit to the Area of Interest
		second_pass_duration: int
			duration spent on the second revisit of the Area of Interest

		"""

		first_pass_duration = 0
		second_pass_duration = 0
		num_readings = 0
		flag = 0


		fix_num, fix_ind, fix_cnt = np.unique(self.data["FixationSeq"], return_index=True, return_counts=True)
		fixation_count = len(fix_num) - 1

		if fixation_count != 0:
			temp1 = []
			fix_ind_end = np.array(fix_ind[1:]) + np.array(fix_cnt[1:])
			for start, end in zip(fix_ind[1:], fix_ind_end):
				if len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)) and flag==0:
					num_readings += 1
					flag = 1

					if num_readings == 1:
						first_pass_duration = end - start
					elif num_readings == 2:
						second_pass_duration = end - start
					elif num_readings == 3:
						break
				elif len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)) and flag==1:
					if num_readings == 1:
						first_pass_duration = end - start
					elif num_readings == 2:
						second_pass_duration = end - start
				elif len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) < int(0.9 * (end-start)):
					flag = 0

				# if len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)) and flag==1:
				# 	first_pass_duration = first_pass_duration + end - start
				# elif flag==1 and first_pass_duration>0:
				# 	flag = 2
				# 	break
				# elif len(np.where(self.data["GazeAOI"][start:end] == 1)[0]) > int(0.9 * (end-start)) and flag==2:
				# 	second_pass_duration = second_pass_duration + end - start
				# elif flag==2 and second_pass_duration>0:
				# 	break

		return first_pass_duration, second_pass_duration


	def findEyeMetaData(self, sampling_freq=1000):
		"""Function to find all metadata/features of eye tracking data.

		Internal function of class that uses its `data` member variable. Can be invoked by an object of the class. The metadata is stored in the sensor object of the class and can be accessed in the following manner.

		Examples
		--------
		The following code will return the metadata dictionary containing all meta features extracted.

		>>> stim_obj.findEyeMetaData()
		>>> stim_obj.sensors["EyeTracker"].metadata

		This segment allows you to extract individual features

		>>> stim_obj.sensors["EyeTracker"].metadata["pupil_slope"]
		>>> stim_obj.sensors["EyeTracker"].metadata["fixation_count"]

		"""

		self.response_time = self.findResponseTime()
		self.sensors["EyeTracker"].metadata["response_time"] = self.response_time

		# Pupil Features
		pupil_size, peak_pupil, time_to_peak, pupil_AUC, pupil_slope, pupil_mean, pupil_size_downsample = self.findPupilParams()
		self.sensors["EyeTracker"].metadata["pupil_size"] = pupil_size
		self.sensors["EyeTracker"].metadata["peak_pupil"] = peak_pupil
		self.sensors["EyeTracker"].metadata["time_to_peak_pupil"] = time_to_peak
		self.sensors["EyeTracker"].metadata["pupil_area_curve"] = pupil_AUC
		self.sensors["EyeTracker"].metadata["pupil_slope"] = pupil_slope
		self.sensors["EyeTracker"].metadata["pupil_mean"] = pupil_mean
		self.sensors["EyeTracker"].metadata["pupil_size_downsample"] = pupil_size_downsample

		# Blink Features
		blink_cnt, peak_blink, avg_blink = self.findBlinkParams()
		self.sensors["EyeTracker"].metadata["blink_rate"] = blink_cnt / self.response_time
		self.sensors["EyeTracker"].metadata["peak_blink_duration"] = peak_blink
		self.sensors["EyeTracker"].metadata["avg_blink_duration"] = avg_blink

		# Fixation Features
		fix_cnt, max_fix_cnt, avg_fix_cnt = self.findFixationParams()
		self.sensors["EyeTracker"].metadata["fixation_count"] = fix_cnt
		self.sensors["EyeTracker"].metadata["max_fixation_duration"] = max_fix_cnt
		self.sensors["EyeTracker"].metadata["avg_fixation_duration"] = avg_fix_cnt

		# Saccade Features
		saccade_count, saccade_duration, saccade_peak_vel, saccade_amplitude = self.findSaccadeParams(sampling_freq)
		self.sensors["EyeTracker"].metadata["sacc_count"] = saccade_count
		self.sensors["EyeTracker"].metadata["sacc_duration"] = saccade_duration
		self.sensors["EyeTracker"].metadata["sacc_vel"] = saccade_peak_vel
		self.sensors["EyeTracker"].metadata["sacc_amplitude"] = saccade_amplitude

		# Microsaccade Features
		_, ms_count, ms_duration, ms_vel, ms_amp = self.findMicrosaccades()
		self.sensors["EyeTracker"].metadata["ms_count"] = ms_count
		self.sensors["EyeTracker"].metadata["ms_duration"] = ms_duration
		self.sensors["EyeTracker"].metadata["ms_vel"] = ms_vel
		self.sensors["EyeTracker"].metadata["ms_amplitude"] = ms_amp

		# ROI Features

		num_revisits = self.numberRevisits()
		first_pass_duration,second_pass_duration = self.passDurationCalculation()

		self.sensors["EyeTracker"].metadata["num_revisits"] = num_revisits
		self.sensors["EyeTracker"].metadata["first_pass_duration"] = first_pass_duration
		self.sensors["EyeTracker"].metadata["second_pass_duration"] = second_pass_duration


	def getData(self, data, sensor_names):
		"""Function to extract data and store in local format.

		It is invoked by `__init__` when the object of the class is created. This function is used in the *Experiment Design*.

		Parameters
		----------
		data : pandas DataFrame
			DataFrame containing the eye tracking data.
		sensor_names : list (str)
			List of sensors being used for the experiment (currently supports only EyeTracker).

		Returns
		-------
		extracted_data : dict
			Dictionary of extracted data to be used by the functions of the class.
			- "ETRows" : list,
			- "FixationSeq" : list,
			- "Gaze" : dict,
			- "InterpPupilSize" : list,
			- "InterpGaze" : dict,
			- "BlinksLeft" : dict,
			- "BlinksRight" : dict

		"""

		# Extracting data for particular stimulus

		with open(self.json_file) as jf:
			contents = json.load(jf)

		columns_of_interest = contents["Columns_of_interest"]["EyeTracker"]

		extracted_data = {	"ETRows" : None,
							"FixationSeq" : None,
							"Gaze" : None,
							"InterpPupilSize" : None,
							"InterpGaze" : None,
							"BlinksLeft" : None,
							"BlinksRight" : None,
							"GazeAOI" : None}

		for col_class in sensor_names:
			if col_class == "EyeTracker":
				et_sfreq = contents["Analysis_Params"]["EyeTracker"]["Sampling_Freq"]

				a = datetime.now()
				self.sensors.update({col_class : Sensor(col_class, et_sfreq)})
				l_gazex_df = np.array(data.GazeLeftx)
				l_gazey_df = np.array(data.GazeLefty)
				r_gazex_df = np.array(data.GazeRightx)
				r_gazey_df = np.array(data.GazeRighty)
				pupil_size_l_df = np.array(data.PupilLeft)
				pupil_size_r_df = np.array(data.PupilRight)
				gaze_aoi_df = np.array(data.GazeAOI)

				# Extracting fixation sequences
				et_rows = np.where(data.EventSource.str.contains("ET"))[0]
				fixation_seq_df = np.array(data.FixationSeq.fillna(-1), dtype='float32')
				fixation_seq = np.squeeze(np.array([fixation_seq_df[i] for i in sorted(et_rows)], dtype="float32"))

				# Extracting the eye gaze data
				l_gaze_x = np.squeeze(np.array([l_gazex_df[i] for i in sorted(et_rows)], dtype="float32"))
				l_gaze_y = np.squeeze(np.array([l_gazey_df[i] for i in sorted(et_rows)], dtype="float32"))
				l_gaze = {"x": l_gaze_x, "y": l_gaze_y}
				r_gaze_x = np.squeeze(np.array([r_gazex_df[i] for i in sorted(et_rows)], dtype="float32"))
				r_gaze_y = np.squeeze(np.array([r_gazey_df[i] for i in sorted(et_rows)], dtype="float32"))
				r_gaze = {"x": r_gaze_x, "y": r_gaze_y}
				gaze = {"left" : l_gaze, "right" : r_gaze}

				# Extracting Pupil Size Data
				pupil_size_r = np.squeeze(np.array([pupil_size_r_df[i] for i in sorted(et_rows)], dtype="float32"))
				pupil_size_l = np.squeeze(np.array([pupil_size_l_df[i] for i in sorted(et_rows)], dtype="float32"))

				# Fixing Blinks and interpolating pupil size and gaze data
				blinks_l, interp_pupil_size_l, new_gaze_l = self.findBlinks(pupil_size_l, gaze=gaze, interpolate=True, concat=True)
				blinks_r, interp_pupil_size_r, new_gaze_r = self.findBlinks(pupil_size_r, gaze=gaze, interpolate=True, concat=True)
				interp_pupil_size = np.mean([interp_pupil_size_r, interp_pupil_size_l], axis=0)

				gaze_aoi = self.setAOICol([gaze_aoi_df, new_gaze_l["left"]["x"], new_gaze_l["left"]["y"]])

				extracted_data["ETRows"] = et_rows
				extracted_data["FixationSeq"] = fixation_seq
				extracted_data["Gaze"] = gaze
				extracted_data["InterpPupilSize"] = interp_pupil_size
				extracted_data["InterpGaze"] = new_gaze_l
				extracted_data["BlinksLeft"] = blinks_l
				extracted_data["BlinksRight"] = blinks_r
				extracted_data["GazeAOI"] = gaze_aoi

		return extracted_data


	def getDataStandAlone(self, data, sensor_names):
		"""Function to extract data and store in local format.

		It is invoked by `__init__` when the object of the class is created. This function is used in the *Stand-alone Design*.

		Parameters
		----------
		data : pandas DataFrame
			DataFrame containing the eye tracking data.
		sensor_names : dict
			Dictionary of dictionaries containing list of sensors being used for the experiment (currently supports only EyeTracker) and their Parameters. See `sensor_names` in Stimulus for details.

		Returns
		-------
		extracted_data : dict
			Dictionary of extracted data to be used by the functions of the class.
			- "ETRows" : list,
			- "FixationSeq" : list,
			- "Gaze" : dict,
			- "InterpPupilSize" : list,
			- "InterpGaze" : dict,
			- "BlinksLeft" : dict,
			- "BlinksRight" : dict

		"""

		extracted_data = {	"ETRows" : None,
							"FixationSeq" : None,
							"Gaze" : None,
							"InterpPupilSize" : None,
							"InterpGaze" : None,
							"BlinksLeft" : None,
							"BlinksRight" : None,
							"GazeAOI" : None}

		gaze_aoi_flag = 1

		for sen in sensor_names:
			if sen == "EyeTracker":
				et_sfreq = sensor_names[sen]["Sampling_Freq"]

				self.sensors.update({sen : Sensor(sen, et_sfreq)})

				data = data[self.start_time : self.end_time]

				l_gazex_df = np.array(data.GazeLeftx)
				l_gazey_df = np.array(data.GazeLefty)
				r_gazex_df = np.array(data.GazeRightx)
				r_gazey_df = np.array(data.GazeRighty)
				pupil_size_l_df = np.array(data.PupilLeft)
				pupil_size_r_df = np.array(data.PupilRight)
				gaze_aoi_df = np.array(data.GazeAOI)

				# Extracting fixation sequences
				et_rows = np.where(data.EventSource.str.contains("ET"))[0]
				fixation_seq_df = np.array(data.FixationSeq.fillna(-1), dtype='float32')
				fixation_seq = np.squeeze(np.array([fixation_seq_df[i] for i in sorted(et_rows)], dtype="float32"))

				# Extracting the eye gaze data
				l_gaze_x = np.squeeze(np.array([l_gazex_df[i] for i in sorted(et_rows)], dtype="float32"))
				l_gaze_y = np.squeeze(np.array([l_gazey_df[i] for i in sorted(et_rows)], dtype="float32"))
				l_gaze = {"x": l_gaze_x, "y": l_gaze_y}
				r_gaze_x = np.squeeze(np.array([r_gazex_df[i] for i in sorted(et_rows)], dtype="float32"))
				r_gaze_y = np.squeeze(np.array([r_gazey_df[i] for i in sorted(et_rows)], dtype="float32"))
				r_gaze = {"x": r_gaze_x, "y": r_gaze_y}
				gaze = {"left" : l_gaze, "right" : r_gaze}

				# Extracting Pupil Size Data
				pupil_size_r = np.squeeze(np.array([pupil_size_r_df[i] for i in sorted(et_rows)], dtype="float32"))
				pupil_size_l = np.squeeze(np.array([pupil_size_l_df[i] for i in sorted(et_rows)], dtype="float32"))

				# Fixing Blinks and interpolating pupil size and gaze data
				blinks_l, interp_pupil_size_l, new_gaze_l = self.findBlinks(pupil_size_l, gaze=gaze, interpolate=True, concat=True)
				blinks_r, interp_pupil_size_r, new_gaze_r = self.findBlinks(pupil_size_r, gaze=gaze, interpolate=True, concat=True)
				interp_pupil_size = np.mean([interp_pupil_size_r, interp_pupil_size_l], axis=0)

				gaze_aoi = self.setAOICol([gaze_aoi_df, new_gaze_l["left"]["x"], new_gaze_l["left"]["y"]])

				extracted_data["ETRows"] = et_rows
				extracted_data["FixationSeq"] = fixation_seq
				extracted_data["Gaze"] = gaze
				extracted_data["InterpPupilSize"] = interp_pupil_size
				extracted_data["InterpGaze"] = new_gaze_l
				extracted_data["BlinksLeft"] = blinks_l
				extracted_data["BlinksRight"] = blinks_r
				extracted_data["GazeAOI"] = gaze_aoi

		return extracted_data


	def setAOICol(self, data):
		"""Function to set values based on a point being inside or outsode the AOI.

		Parameters
		---------
		data : list
			List of size 3 containing gaze_aoi, gaze_x and gaze_y column data.

		Returns
		-------
		gaze_aoi_new : list
			Modified gaze_aoi column with the mask for points inside and outside the AOI.

		"""
		patch = None
		# Rectangle AOI

		if len(self.aoi_coords) == 4 and (isinstance(self.aoi_coords[0], float) or isinstance(self.aoi_coords[0], int)):
			patch = Rectangle((self.aoi_coords[0], self.aoi_coords[1]),
									(self.aoi_coords[2] - self.aoi_coords[0]),
									(self.aoi_coords[3] - self.aoi_coords[1]),
									color='r', fill=False, linestyle='--')

		# Circle AOI
		elif len(self.aoi_coords) == 3:
			patch = Ellipse((self.aoi_coords[0][0], self.aoi_coords[0][1]),
								self.aoi_coords[1],
								self.aoi_coords[2],
								color='r', fill=False, linestyle='--')

		# Polygon AOI
		else:
			xy = np.asarray(self.aoi_coords)
			patch = Polygon(xy, color='r', fill=False, linestyle='--')

		gaze_aoi = data[0]
		x = data[1]
		y = data[2]

		points = np.transpose(np.vstack((x, y)))
		contains = patch.contains_points(points)
		gaze_aoi_new = np.asarray(contains, dtype=int)
		gaze_aoi_new[np.where(gaze_aoi_new == 0)[0]] = -1

		return gaze_aoi_new


def groupHeatMap(sub_list, stim_name, json_file, save_fig=False):
	"""Function to plot aggregate heat map of gaze for a list if subjects.

		Invoked by the `subjectVisualize <#Subject.Subject.subjectVisualize>`_ function of the `Subject <#module-Subject>`_ class.

		Parameters
		----------
		sub_list : list (Subject)
			List of `Subject` class objects to plot the gaze heat map for.
		stim_name : dict
			Dictionary containing the type of stimulus and the number of stimulus of that type. {stim_type:stim_num}
		json_file : str
			Name of json file containing details of the experiment.
		save_fig : bool
			Save the figure or not.

	"""

	fig = plt.figure()
	fig.canvas.set_window_title("Aggregate Gaze Heat Map")

	ax = plt.gca()

	x = []
	y = []
	stim_type, stim_num = stim_name.popitem()

	cnt = 0
	for sub in sub_list:
		if sub.stimulus[stim_type][stim_num].data != None:
			x = np.concatenate((x, sub.stimulus[stim_type][stim_num].data["InterpGaze"]["left"]["x"]))
			y = np.concatenate((y, sub.stimulus[stim_type][stim_num].data["InterpGaze"]["left"]["y"]))
		else:
			cnt += 1

	if cnt == len(sub_list):
		plt.close(fig)
		return

	# In order to get more intense values for the heatmap (ratio of points is unaffected)
	x = np.repeat(x, 5)
	y = np.repeat(y, 5)

	with open(json_file) as json_f:
		json_data = json.load(json_f)
		path = json_data["Path"]
		path = path.replace("\\", "/")
		width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
		height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]
		aoi_coords = json_data["Analysis_Params"]["EyeTracker"]["aoi"]

	try:
		img = plt.imread(path + "/Stimuli/" + sub_list[0].stimulus[stim_type][stim_num].name + ".jpg")
	except:
		try:
			img = plt.imread(path + "Stimuli/" + sub_list[0].stimulus[stim_type][stim_num].name + ".jpeg")
		except:
			img = np.zeros((height, width))


	downsample_fraction = 0.25
	col_shape = img.shape[1]
	row_shape = img.shape[0]

	hist, _, _ = np.histogram2d(y, x, bins=[int(col_shape*downsample_fraction), int(row_shape*downsample_fraction)], range=[[0, int(col_shape)], [0, int(row_shape)]])
	hist = gaussian_filter(hist, sigma=12)
	hist = hist.repeat(int(1/downsample_fraction), axis=0).repeat(int(1/downsample_fraction), axis=1)

	cmap = pl.cm.jet
	my_cmap = cmap(np.arange(cmap.N))
	my_cmap[:cmap.N//4, -1] = 0
	my_cmap[cmap.N//4:, -1] = np.linspace(0.2, 0.4, cmap.N - cmap.N//4)
	my_cmap = ListedColormap(my_cmap)

	ax.imshow(img)

	# Rectangle AOI
	if len(aoi_coords) == 4 and (isinstance(aoi_coords[0], float) or isinstance(aoi_coords[0], int)):
		rect = Rectangle((aoi_coords[0], aoi_coords[1]),
								(aoi_coords[2]-aoi_coords[0]),
								(aoi_coords[3]-aoi_coords[1]),
										color='r', fill=False, linestyle='--')
		ax.add_patch(rect)

	# Circle AOI
	elif len(aoi_coords) == 3:
		ellipse = Ellipse((aoi_coords[0][0], aoi_coords[0][1]),
								(aoi_coords[1]),
								(aoi_coords[2]),
										color='r', fill=False, linestyle='--')
		ax.add_patch(ellipse)

	# Polygon AOI
	else:
		xy = np.asarray(aoi_coords)
		poly = Polygon(xy, color='r', fill=False, linestyle='--')
		ax.add_patch(poly)

	ax.contourf(np.arange(0, int(row_shape), 1), np.arange(0, int(col_shape), 1), hist, cmap=my_cmap)
	ax.set_xlim(0, int(col_shape))
	ax.set_ylim(int(row_shape), 0)

	if save_fig:
		if not os.path.isdir(path + "/Aggregate_Plots/"):
			os.mkdir(path + "/Aggregate_Plots/")
		fig.savefig(path + "/Aggregate_Plots"+ "/agg_gaze_heatmap_" + str(datetime.now().timestamp()).split(".")[0] + ".png", dpi=300)
		with open(path + "/Aggregate_Plots"+ "/agg_gaze_heatmap_" + str(datetime.now().timestamp()).split(".")[0] + ".txt", "w") as f:
			for sub in sub_list:
				f.write(sub.name + "\n")

	plt.show()
	plt.close(fig)