import os
import json
import numpy as np
import pandas as pd
from scipy import signal, io, stats, misc
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.animation as animation
from Sensor import Sensor
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

import matplotlib as mpl
mpl.use("TkAgg")


class Stimulus:

	def __init__(self, name="id_rather_not", stim_type="doesnt_matter", sensor_names=["EyeTracker"], data=None, start_time=-1, end_time=-1, roi_time=-1, json_file=None, subject_name="buttersnaps"):
		self.name = name
		self.stim_type = stim_type
		self.start_time = start_time
		self.end_time = end_time
		self.response_time = self.end_time - self.start_time
		self.roi_time = roi_time
		self.json_file = json_file
		self.sensors = dict()
		self.subject_name = subject_name
		
		if not os.path.isdir('./Subjects/' + self.subject_name + '/'):
			os.makedirs('./Subjects/' + self.subject_name + '/')

		# Experiment json file exists so stimulus is being created for experiment
		if self.json_file != None:
			if self.start_time == -1:
				self.data = None
			else:
				self.data = self.getData(data, sensor_names)
		
		# Experiment json file does not exist so stimulus is being created as a stand alone object
		else:
			self.data = self.getDataStandAlone(data, sensor_names)


	def diff(self, series):
		"""
		Python implementation of Matlab's 'diff' function. Returns (a[n+1] - a[n]) for all n.
		"""
		return series[1:] - series[:-1]


	def smooth(self, x, window_len):
		"""
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
		"""
		Function to find blinks and return blink onset, offset indices and interpolated pupil size data
		Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,” Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.
		
		Parameters
		----------

		Input:
			pupil_size          : [numpy array/list] of average pupil size data for left and right eye
			gaze                : [dictionary] {"x", "y"} containing numpy array/list of gaze in x and y direction
			sampling_freq       : [float] sampling frequency of eye tracking hardware (default = 1000 Hz)
			concat              : [boolean] concatenate close blinks/missing trials or not. See R. Hershman et. al. for more information 
			concat_gap_interval : [float] interval between successive missing samples/blinks to concatenate
		Output:	
			blinks              : [dictionary] {"blink_onset", "blink_offset"} containing numpy array/list of blink onset and offset indices
			interp_pupil_size   : [numpy array/list] of interpolated average pupil size data for left and right eye after fixing blinks 
			new_gaze            : [dictionary] {"x", "y"} containing numpy array/list of interpolated gaze in x and y direction after fixing blinks
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

					# if eye == "left":
					# 	plt.plot(gaze_x, gaze_y)

					interp_gaze_x = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices), gaze_x_no_blinks)
					interp_gaze_y = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices), gaze_y_no_blinks)

					# if eye == "left":
					# 	plt.plot(interp_gaze_x, interp_gaze_y, alpha=0.5)

					# plt.show()
					new_gaze[eye] = {"x": interp_gaze_x, "y": interp_gaze_y}

		else:
			interp_pupil_size = pupil_size
			new_gaze = gaze

		return blinks, interp_pupil_size, new_gaze


	def findFixations(self):
		"""
		Function to extract fixation sequences from iMotions data
		Input:
			fixation_seq     : [numpy array] of fixation sequences identified by Duration Dispersion
		Ouput:
			fixation_indices : [dictionary] {"start", "end"} of numpy arrays containing the indices of start and end of fixations
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
		"""
		Function to extract fixation sequences from iMotions data
		Input:
			fixation_seq     : [numpy array] of fixation sequences identified by Duration Dispersion
		Ouput:
			fixation_indices : [dictionary] {"start", "end"} of numpy arrays containing the indices of start and end of fixations
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
		"""
		Function to calculate velocity for a point based on 6 samples 
		Input:
			gaze          : [numpy array] of gaze positons in x or y direction
			sampling_freq : float
		Output:
			velocity      : [numpy array] of gaze velocities in x or y direction
		"""
	
		n = len(gaze)
		velocity = np.zeros(gaze.shape)
	
		velocity[2:n-2] = (gaze[4:n] + gaze[3:n-1] - gaze[1:n-3] - gaze[0:n-4]) * (sampling_freq/6.0)
		velocity[1] = (gaze[2] - gaze[0]) * (sampling_freq/2.0)
		velocity[n-2] = (gaze[n-1] - gaze[n-3]) * (sampling_freq/2.0)
	
		return velocity


	def smoothGaze(self, vel, gaze, sampling_freq):
		"""
		Function to smoothen gaze positions using running average method
		Input:
			vel           : [numpy array] of gaze velocities in x or y direction
			gaze          : [numpy array] of gaze positons in x or y direction
			sampling_freq : float
		Output:
			smooth_gaze   : [numpy array] of smoothened gaze positons in x or y direction
		"""
	
		smooth_gaze = np.zeros(gaze.shape)
	
		smooth_gaze[0] = gaze[0]
		vel[1] = vel[1] + smooth_gaze[0]
		smooth_gaze = np.cumsum(vel)
	
		return smooth_gaze


	def calculateMSThreshold(self, vel, sampling_freq, VFAC=5.0):
		"""
		Function to calculate threshold value for X and Y directions
	
		Input:
			vel           : [numpy array] gaze velocity in x and y direction 
			sampling_freq : [float] sampling frequency of the eye tracking device
			VFAC          : [float] 
		Output:
			radius        : [float] threshold radius in x or y direction	
		"""
	
		medx = np.median(vel)
		msdx = np.sqrt(np.median((vel-medx)**2))
	
		if msdx<1e-10:
			msdx = np.sqrt(np.mean(vel**2) - (np.mean(vel))**2)
	
		radius = VFAC * msdx
		return radius


	def findBinocularMS(self, msl, msr):
		"""
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
			# print(m)

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
		"""
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

		ms_count = 0
		ms_duration = []

		ms_count = len(MS)
		for ms in MS:
			ms_duration.append(ms[1] - ms[0])
		return np.array(MS), ms_count, ms_duration


	def findMicrosaccades(self, sampling_freq=1000, plot_ms=False):
		"""
		Function to detect microsaccades within fixations.
		Adapted from R. Engbert and K. Mergenthaler, “Microsaccades are triggered by low retinal image slip,” Proc. Natl. Acad. Sci., vol. 103, no. 18, pp. 7192–7197, 2006.

		Input:

		Output:

		"""
		fixation_indices = self.findFixations()
		all_bin_MS = []

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
			

			if plot_ms:
				
				# Plot gaze and velocity with thresholds
				fig = plt.figure()
				a1 = fig.add_subplot(1, 2, 1)
				a2 = fig.add_subplot(1, 2, 2)

				a1.plot(smooth_gaze["left"]["x"][1:], smooth_gaze["left"]["y"][1:])
				a1.set_xlabel("x")
				a1.set_ylabel("y")
				a1.set_title("gaze plot")
				for i in range(len(MS["bin"])):
					a1.plot(smooth_gaze["left"]["x"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], smooth_gaze["left"]["y"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], color='r')

				e = Ellipse((0, 0), 2*MS["bin"][0, 7], 2*MS["bin"][0, 8], linestyle='--', color='g', fill=False)
				a2.add_artist(e)

				a2.plot(vel["left"]["x"], vel["left"]["y"], alpha=0.5)
				a2.set_xlabel("vel-x")
				a2.set_ylabel("vel-y")
				a2.set_title("gaze velocity plot")
				for i in range(len(MS["bin"])):
					a2.plot(vel["left"]["x"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], vel["left"]["y"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], color='r') 

				plt.savefig("./Subjects/" + self.subject_name + "/ms_gaze_vel" + self.name + "_" + str(fix_ind) + ".png")
				fig.close()


			if plot_ms:
				# Plot main sequence i.e peak velocity vs peak amplitude
				fig2 = plt.figure()
				plt.xlabel("Amplitude (deg)")
				plt.ylabel("Peak Velocity (deg/s)")
				for ms in all_bin_MS:
					for i in range(len(ms["bin"])):
						peak_vel = (ms["bin"][i, 2] + ms["bin"][i, 11])/2
						amp = (np.sqrt(ms["bin"][i, 5]**2 + ms["bin"][i, 6]**2) + np.sqrt(ms["bin"][i, 13]**2 + ms["bin"][i, 14]**2))/2

						plt.scatter(amp, peak_vel, marker='o', facecolors='none', edgecolors='r')
				
				plt.savefig("./Subjects/" + self.subject_name + "/ms_main_seq" + self.name + ".png")
				fig2.close()

		ms_count = 0
		ms_duration = np.zeros(1, dtype='float32')
		temp_vel = np.zeros(1, dtype='float32')
		temp_amp = np.zeros(1, dtype='float32')
		for ms in all_bin_MS:
			# Net microsaccade count i.e binary + left +right
			ms_count += ms["NB"] + ms["NL"] + ms["NR"]

			if(ms["NB"] != 0):
				# Appending peak velocity for binary microsaccade
				vel_val = (ms["bin"][:, 2] + ms["bin"][:, 11]) / 2.
				temp_vel = np.hstack((temp_vel, vel_val))
			
				# Appending amplitude for binary microsaccade
				amp_val = (np.sqrt(ms["bin"][:, 5]**2 + ms["bin"][:, 6]**2) + np.sqrt(ms["bin"][:, 13]**2 + ms["bin"][:, 14]**2)) / 2.
				temp_amp = np.hstack((temp_amp, amp_val))

				# Appending durations for binary microsaccade
				dur_val = ((ms["bin"][:, 1] - ms["bin"][:, 0]) + (ms["bin"][:, 10] - ms["bin"][:, 9])) / 2.
				ms_duration = np.hstack((ms_duration, dur_val))	
			
			if(ms["NL"] != 0):
				# Appending peak velocity for left eye microsaccade
				temp_vel = np.hstack((temp_vel, ms["left"][:, 2]))
				
				# Appending amplitude for left eye microsaccade
				temp_amp = np.hstack((temp_amp, np.sqrt(ms["left"][:, 5]**2 + ms["left"][:, 6]**2)))

				# Appending durations for left eye microsaccade
				dur_val = ms["left"][:, 1] - ms["left"][:, 0]
				ms_duration = np.hstack((ms_duration, dur_val))
			
			if(ms["NR"] != 0):
				# Appending peak velocity for right eye microsaccade
				temp_vel = np.hstack((temp_vel, ms["right"][:, 2]))
				
				# Appending amplitude for right eye microsaccade
				temp_amp = np.hstack((temp_amp, np.sqrt(ms["right"][:, 5]**2 + ms["right"][:, 6]**2)))

				# Appending durations for right eye microsaccade
				dur_val = ms["right"][:, 1] - ms["right"][:, 0]
				ms_duration = np.hstack((ms_duration, dur_val))

		if ms_count == 0:
			ms_duration = [0, 0]
			temp_vel = [0, 0]
			temp_amp = [0, 0]
			
		return all_bin_MS, ms_count, ms_duration[1:], temp_vel[1:], temp_amp[1:]


	def findSaccadeParams(self, sampling_freq=1000):
		"""
		"""
		
		saccade_indices = self.findSaccades()
		saccade_onset = saccade_indices["start"]
		saccade_offset = saccade_indices["end"]

		saccade_count = 0
		saccade_duration = np.array(saccade_offset) - np.array(saccade_onset)
		
		saccade_peak_vel = []
		saccade_amplitude = []

		for start, end in zip(saccade_onset, saccade_offset):
			if (end-start) < 6:
				continue

			saccade_count += 1

			vel_x = self.position2Velocity(self.data["Gaze"]["left"]["x"][start:end], sampling_freq)
			vel_y = self.position2Velocity(self.data["Gaze"]["left"]["y"][start:end], sampling_freq)

			# Saccade peak velocity (vpeak)
			vpeak = max(np.sqrt(vel_x**2 + vel_y**2))
			saccade_peak_vel.append(vpeak)

			# Saccade amplitude (dX,dY)
			minx = min(self.data["Gaze"]["left"]["x"][start:end])
			maxx = max(self.data["Gaze"]["left"]["x"][start:end])
			miny = min(self.data["Gaze"]["left"]["y"][start:end])
			maxy = max(self.data["Gaze"]["left"]["y"][start:end])
			ix1 = np.argmin(self.data["Gaze"]["left"]["x"][start:end])
			ix2 = np.argmax(self.data["Gaze"]["left"]["x"][start:end])
			iy1 = np.argmin(self.data["Gaze"]["left"]["y"][start:end])
			iy2 = np.argmax(self.data["Gaze"]["left"]["y"][start:end])
			dX = np.sign(ix2 - ix1) * (maxx - minx)
			dY = np.sign(iy2 - iy1) * (maxy - miny)
			saccade_amplitude.append(np.sqrt(dX**2 + dY**2))
		
		return (saccade_count, saccade_duration, saccade_peak_vel, saccade_amplitude)


	def findResponseTime(self, sampling_freq=1000):
		"""
		"""
		return len(self.data["ETRows"] * (1000/sampling_freq))
	

	def findFixationParams(self):
		"""
		"""
		fix_num, fix_cnt = np.unique(self.data["FixationSeq"], return_counts=True)
		
		fixation_count = len(fix_num) - 1

		if fixation_count != 0:
			max_fixation_duration = np.max(fix_cnt[1:])
			avg_fixation_duration = np.mean(fix_cnt[1:])
		else:
			max_fixation_duration = 0
			avg_fixation_duration = 0
		
		return (fixation_count, max_fixation_duration, avg_fixation_duration)


	def findPupilParams(self):
		"""
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
		"""
		"""

		blink_cnt = len(self.data["BlinksLeft"]["blink_onset"])
		if blink_cnt != 0:
			peak_blink_duration = np.max((np.array(self.data["BlinksLeft"]["blink_offset"]) - np.array(self.data["BlinksLeft"]["blink_onset"])))
			avg_blink_duration = np.mean((np.array(self.data["BlinksLeft"]["blink_offset"]) - np.array(self.data["BlinksLeft"]["blink_onset"])))
		else:
			peak_blink_duration = 0
			avg_blink_duration = 0
		
		return (blink_cnt, peak_blink_duration, avg_blink_duration)


	def gazePlot(self, save_fig=False, show_fig=True):
		"""
		"""
		
		if self.data == None:
			return

		fig = plt.figure()
		fig.canvas.set_window_title("Gaze Plot: " + self.name)

		try:
			img = plt.imread("Stimuli/" + self.name + ".jpg")
		except:
			try:
				img = plt.imread("Stimuli/" + self.name + ".jpeg")
			except:
				with open(self.json_file) as json_f:
					json_data = json.load(json_f)
				width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
				height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]
				img = np.zeros((height, width))
		
		ax = plt.gca()
		ax.imshow(img)

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
		for x, y in zip(fixation_gaze_x, fixation_gaze_y):
			ax.plot(np.mean(x), np.mean(y), 'go', markersize=15, alpha=0.7)
			ax.text(np.mean(x), np.mean(y), str(i), fontsize=10, color='w')
			i += 1

		if show_fig:
			plt.show()
		
		if save_fig:
			fig.savefig("./Subjects/" + self.subject_name + "/gaze_plot_" + self.name + ".png", dpi=300)


	def gazeHeatMap(self, save_fig=False, show_fig=True):
		"""
		"""
		
		if self.data == None:
			return

		fig = plt.figure()
		fig.canvas.set_window_title("Gaze Heat Map: " + self.name)
		
		ax = plt.gca()
		
		x = self.data["InterpGaze"]["left"]["x"]
		y = self.data["InterpGaze"]["left"]["y"]
		
		# In order to get more intense values for the heatmap (ratio of points is unaffected)
		x = np.repeat(x, 5)
		y = np.repeat(y, 5)

		try:
			img = plt.imread("Stimuli/" + self.name + ".jpg")
		except:
			try:
				img = plt.imread("Stimuli/" + self.name + ".jpeg")
			except:
				with open(self.json_file) as json_f:
					json_data = json.load(json_f)
				width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
				height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]
				img = np.zeros((height, width))

		downsample_fraction = 0.25
		col_shape = img.shape[1]
		row_shape = img.shape[0]

		hist, _, _ = np.histogram2d(x, y, bins=[int(row_shape*downsample_fraction), int(col_shape*downsample_fraction)], range=[[0, int(row_shape)],[0, int(col_shape)]])
		hist = gaussian_filter(hist, sigma=12)

		mycmap = cm.GnBu
		mycmap._init()
		mycmap._lut[:,-1] = np.linspace(0, 0.8, 255+4)
		img = misc.imresize(img, size=downsample_fraction, interp='lanczos')
		ax.imshow(img)
		ax.contourf(np.arange(0, int(row_shape*downsample_fraction), 1), np.arange(0, int(col_shape*downsample_fraction), 1), hist.T, cmap=mycmap)
		ax.set_xlim(0, int(col_shape * downsample_fraction))
		ax.set_ylim(int(row_shape * downsample_fraction), 0)

		if show_fig:
			plt.show()
			plt.close(fig)

		if save_fig:
			fig.savefig("./Subjects/" + self.subject_name + "/gaze_heatmap_" + self.name + ".png", dpi=300)


	def visualize(self):
		"""
		Function to create dynamic plot of subject data (gaze, pupil size, eeg(Pz))
		
		Input:
			
		Output:
			NA
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
			img = plt.imread("Stimuli/" + self.name + ".jpg")
		except:
			try:
				img = plt.imread("Stimuli/" + self.name + ".jpeg")
			except:
				with open(self.json_file) as json_f:
					json_data = json.load(json_f)
				width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
				height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]
				img = np.zeros((height, width))
				
		ax.imshow(img)

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
			nonlocal is_manual
			is_manual = True
			val = int(val)
			update(val)

		def update(i):
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
			nonlocal is_manual
			if is_manual:
				return [line, circle, line3]

			i = int(samp.val + 1) % total_range[-1]
			samp.set_val(i)
			is_manual = False # the above line called update_slider, so we need to reset this
			fig.canvas.draw_idle()
			
			return [line, circle, line3]
			

		def on_click(event):
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

		plt.show()


	def findEyeMetaData(self, sampling_freq=1000):
		"""
		Input:
			subject_name : [string] Name of subject to visualize data for 
			stimuli_name : [list] of strings containing stimuli names to visualize the data for
		Output:
			NA
		"""

		# Finding word and character count in text stimulus
		num_chars = 1
		num_words = 1
		if self.stim_type in ["alpha", "relevant", "general", "general_lie"]:
			with open("questions.json") as q_file:
				data = json.load(q_file)

			num_chars = len(data[self.name])
			num_words = len(data[self.name].split())

		# Finding response time based on number of  samples 
		self.response_time = self.findResponseTime()
		self.sensors["EyeTracker"].metadata["response_time"] = self.response_time / num_words
	
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


	def getData(self, data, sensor_names):
		# Extracting data for particular stimulus
		
		with open(self.json_file) as jf:
			contents = json.load(jf)

		extracted_data = {	"ETRows" : None,
							"FixationSeq" : None,
							"Gaze" : None,
							"InterpPupilSize" : None,
							"InterpGaze" : None,
							"BlinksLeft" : None,
							"BlinksRight" : None}

		for col_class in sensor_names:
			if col_class == "EyeTracker":
				et_sfreq = contents["Analysis_Params"]["EyeTracker"]["Sampling_Freq"]

				self.sensors.update({col_class : Sensor(col_class, et_sfreq)})

				l_gazex_df = np.array(data.GazeLeftx)
				l_gazey_df = np.array(data.GazeLefty)
				r_gazex_df = np.array(data.GazeRightx)
				r_gazey_df = np.array(data.GazeRighty)
				pupil_size_l_df = np.array(data.PupilLeft)
				pupil_size_r_df = np.array(data.PupilRight)

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

				extracted_data["ETRows"] = et_rows
				extracted_data["FixationSeq"] = fixation_seq
				extracted_data["Gaze"] = gaze
				extracted_data["InterpPupilSize"] = interp_pupil_size
				extracted_data["InterpGaze"] = new_gaze_l
				extracted_data["BlinksLeft"] = blinks_l
				extracted_data["BlinksRight"] = blinks_r

		return extracted_data


	def getDataStandAlone(self, data, sensor_names):
		"""
		Function to create data for stand alone data file and not entire experiment
		"""

		extracted_data = {	"ETRows" : None,
							"FixationSeq" : None,
							"Gaze" : None,
							"InterpPupilSize" : None,
							"InterpGaze" : None,
							"BlinksLeft" : None,
							"BlinksRight" : None}

		for sen in sensor_names:
			if sen == "EyeTracker":
				et_sfreq = sensor_names[sen]["Sampling_Freq"]

				self.sensors.update({sen : Sensor(sen, et_sfreq)})

				l_gazex_df = np.array(data.GazeLeftx)
				l_gazey_df = np.array(data.GazeLefty)
				r_gazex_df = np.array(data.GazeRightx)
				r_gazey_df = np.array(data.GazeRighty)
				pupil_size_l_df = np.array(data.PupilLeft)
				pupil_size_r_df = np.array(data.PupilRight)

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

				extracted_data["ETRows"] = et_rows
				extracted_data["FixationSeq"] = fixation_seq
				extracted_data["Gaze"] = gaze
				extracted_data["InterpPupilSize"] = interp_pupil_size
				extracted_data["InterpGaze"] = new_gaze_l
				extracted_data["BlinksLeft"] = blinks_l
				extracted_data["BlinksRight"] = blinks_r

		return extracted_data


def groupHeatMap(sub_list, stim_name, json_file):
	"""
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

	try:
		img = plt.imread("Stimuli/" + sub_list[0].stimulus[stim_type][stim_num].name + ".jpg")
	except:
		try:
			img = plt.imread("Stimuli/" + sub_list[0].stimulus[stim_type][stim_num].name + ".jpeg")
		except:
			with open(json_file) as json_f:
				json_data = json.load(json_f)
			width = json_data["Analysis_Params"]["EyeTracker"]["Display_width"]
			height = json_data["Analysis_Params"]["EyeTracker"]["Display_height"]
			img = np.zeros((height, width))

	downsample_fraction = 0.25
	col_shape = img.shape[1]
	row_shape = img.shape[0]

	hist, _, _ = np.histogram2d(x, y, bins=[int(row_shape*downsample_fraction), int(col_shape*downsample_fraction)], range=[[0, int(row_shape)],[0, int(col_shape)]])
	hist = gaussian_filter(hist, sigma=12)

	mycmap = cm.GnBu
	mycmap._init()
	mycmap._lut[:,-1] = np.linspace(0, 0.8, 255+4)
	img = misc.imresize(img, size=downsample_fraction, interp='lanczos')
	ax.imshow(img)
	ax.contourf(np.arange(0, int(row_shape*downsample_fraction), 1), np.arange(0, int(col_shape*downsample_fraction), 1), hist.T, cmap=mycmap)
	ax.set_xlim(0, int(col_shape * downsample_fraction))
	ax.set_ylim(int(row_shape * downsample_fraction), 0)

	plt.show()
	plt.close(fig)