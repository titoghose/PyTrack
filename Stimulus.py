import json
import numpy as np
import pandas as pd
from scipy import signal, io
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.animation as animation
from Sensor import Sensor

class Stimulus:

	def __init__(self, name, stim_type, sensor_names, data, start_time, end_time, roi_time, json_file):
		self.name = name
		self.stim_type = stim_type
		self.start_time = start_time
		self.end_time = end_time
		self.response_time = self.end_time - self.start_time
		self.roi_time = roi_time
		self.json_file = json_file
		self.sensors = []

		if(self.start_time == -1):
			self.data = None
		else:
			self.data = self.getData(data, sensor_names)


	def diff(self, series):
		"""
		Python implementation of Matlab's 'diff' function. Returns (a[n+1] - a[n]) for all n.
		"""
		return series[1:] - series[:-1]


	def smooth(self, x, window_len):
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


	def fix_blinks(self, pupil_size, gaze=None, sampling_freq=1000, concat=False, concat_gap_interval=100, interpolate=False):
		"""
		Function to find blinks and return blink onset, offset indices and interpolated pupil size data
		Adapted from: R. Hershman, A. Henik, and N. Cohen, “A novel blink detection method based on pupillometry noise,” Behav. Res. Methods, vol. 50, no. 1, pp. 107–114, 2018.
	
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

		# starts with a blink
		if len(blink_onset) < len(blink_offset):
			blink_onset.insert(0, 0)
		
		# ends on blink
		if len(blink_onset) > len(blink_offset):
			blink_offset.append(len(difference))

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

		else:
			interp_pupil_size = pupil_size
			new_gaze = gaze

		return blinks, interp_pupil_size, new_gaze


	def findFixationsIDT(self, fixation_seq):
		"""
		Function to extract fixation sequences from iMotions data
		Input:
			fixation_seq     : [numpy array] of fixation sequences identified by Duration Dispersion
		Ouput:
			fixation_indices : [dictionary] {"start", "end"} of numpy arrays containing the indices of start and end of fixations
		"""	
	
		fixation_ind = np.where(fixation_seq != -1)[0]
		fixation_ind_diff = self.diff(fixation_ind)
		indices = np.where(fixation_ind_diff != 1)
	
		fixation_onset = []
		fixation_offset = []
	
		i = 0
	
		while i < len(fixation_seq):
			if fixation_seq[i] != -1:
				curr = fixation_seq[i]
				fixation_onset.append(i)
				while i < len(fixation_seq) and fixation_seq[i] != -1 and fixation_seq[i] == curr:
					i += 1
				fixation_offset.append(i-1)
			else:
				i += 1
	
		fixation_indices = {"start": fixation_onset, "end": fixation_offset}
	
		return fixation_indices


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


	def findMicrosaccades(self, fixation_seq, gaze, sampling_freq=1000):
		"""
		Function to detect microsaccades within fixations.
		Adapted from R. Engbert and K. Mergenthaler, “Microsaccades are triggered by low retinal image slip,” Proc. Natl. Acad. Sci., vol. 103, no. 18, pp. 7192–7197, 2006.

		Input:

		Output:

		"""
		fixation_indices = self.findFixationsIDT(fixation_seq)
		all_bin_MS = []

		for fix_ind in range(len(fixation_indices["start"])):

			all_MS = {"left" : None, "right" : None}
			ms_count = {"left" : None, "right" : None}
			ms_duration = {"left" : None, "right" : None}
			smooth_gaze = {"left" : None, "right" : None}
			vel = {"left" : None, "right" : None}


			for i in ["left", "right"]:

				curr_gaze = {"x" : gaze[i]["x"][fixation_indices["start"][fix_ind] : fixation_indices["end"][fix_ind] + 1],
							"y" : gaze[i]["y"][fixation_indices["start"][fix_ind] : fixation_indices["end"][fix_ind] + 1]}

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
			
			# fig = plt.figure()
			# a1 = fig.add_subplot(1, 2, 1)
			# a2 = fig.add_subplot(1, 2, 2)

			# a1.plot(smooth_gaze["left"]["x"][1:], smooth_gaze["left"]["y"][1:])
			# a1.set_xlabel("x")
			# a1.set_ylabel("y")
			# a1.set_title("gaze plot")
			# for i in range(len(MS["bin"])):
			# 	a1.plot(smooth_gaze["left"]["x"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], smooth_gaze["left"]["y"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], color='r')

			# e = Ellipse((0, 0), 2*MS["bin"][0, 7], 2*MS["bin"][0, 8], linestyle='--', color='g', fill=False)
			# a2.add_artist(e)

			# a2.plot(vel["left"]["x"], vel["left"]["y"], alpha=0.5)
			# a2.set_xlabel("vel-x")
			# a2.set_ylabel("vel-y")
			# a2.set_title("gaze velocity plot")
			# for i in range(len(MS["bin"])):
			# 	a2.plot(vel["left"]["x"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], vel["left"]["y"][int(MS["bin"][i, 0]) : int(MS["bin"][i, 1]) + 1], color='r') 

			# plt.savefig("sampleDataPlot.png")
			# plt.show()

		ms_count = 0
		ms_duration = []
		for ms in all_bin_MS:
			ms_count += ms["NB"] + ms["NL"] + ms["NR"]
			if(ms["NB"] != 0):
				for bms in ms["bin"]:
					ms_duration.append((bms[1] - bms[0]) + (bms[10] - bms[9]) / 2.)
			if(ms["NL"] != 0):
				for lms in ms["left"]:
					ms_duration.append(lms[1] - lms[0])
			if(ms["NR"] != 0):
				for rms in ms["right"]:
					ms_duration.append(rms[1] - rms[0])

		return all_bin_MS, ms_count, ms_duration


	def visualize(self):
		"""
		Function to create dynamic plot of subject data (gaze, pupil size, eeg(Pz))
		
		Input:
			
		Output:
			NA
		"""
		
		if len(self.sensors) == 0:
			return

		total_range = None
		viz_eeg = None
		eeg_lines = None
		eeg_channels = None
		# Initialising Plots
		fig = plt.figure()
		fig.canvas.set_window_title(self.name)
		ax = fig.add_subplot(3, 1, 1)
		ax2 = fig.add_subplot(3, 1, 2)
		ax3 = fig.add_subplot(3, 1, 3)
		
		img = plt.imread("Stimuli/" + self.name + ".jpg")
		ax.imshow(img)

		if self.data["InterpGaze"] != None:
			total_range = range(len(self.data["ETRows"]))
			# Plot for eye gaze
			line, = ax.plot(self.data["InterpGaze"]["left"]["x"][:1], self.data["InterpGaze"]["left"]["y"][:1], 'r-', alpha=1)
			circle, = ax.plot(self.data["InterpGaze"]["left"]["x"][1], self.data["InterpGaze"]["left"]["y"][1], 'go', markersize=10, alpha=0.7)
			ax.set_title("Gaze")
		
			# Plot for pupil size
			line3, = ax3.plot(total_range[:1], self.data["InterpPupilSize"][:1])
			ax3.set_xlim([0, len(total_range)])
			ax3.set_ylim([-2, 11])
			ax3.set_title("Pupil Size vs. Time")
			ax3.set_xlabel("Time (ms)")
			ax3.set_ylabel("Pupil Size")

			for i in range(len(self.data["BlinksLeft"]["blink_onset"])):
				plt.axvline(x=self.data["BlinksLeft"]["blink_onset"][i], linestyle="--", color="r", alpha=0.4)
				plt.axvline(x=self.data["BlinksLeft"]["blink_offset"][i], linestyle="--", color="g", alpha=0.6)
			
				plt.axvline(x=self.data["BlinksLeft"]["blink_onset"][i], linestyle="--", color="r", alpha=0.4)
				plt.axvline(x=self.data["BlinksLeft"]["blink_offset"][i], linestyle="--", color="g", alpha=0.6)

		if self.data["EEG"] != None:
			# Plot for eeg (Pz)
			viz_eeg = self.data["EEG"].copy()
			eeg_channels = [i for i in viz_eeg]

			if total_range != None:
				for channel in viz_eeg:
					(viz_eeg[channel], eeg_time) = signal.resample(viz_eeg[channel], len(total_range), t=sorted(self.data["EEGRows"]))
			else:
				total_range = range(len(self.data["EEGRows"]))
			
			eeg_lines = []
			for ind, channel in enumerate(viz_eeg):
				eeg_lines_temp, = ax2.plot(total_range[:1], viz_eeg[channel][:1], alpha=0.7)
				eeg_lines.append(eeg_lines_temp)
				eeg_lines[ind].set_visible(False)
			eeg_lines[0].set_visible(True)
			
			ax2.set_xlim([0, len(total_range)])
			ax2.set_title("Usampled EEG : [256Hz to 1000Hz] vs. Time")
			ax2.set_xlabel("Time (ms)")
			ax2.set_ylabel("EEG")

		axamp = plt.axes([0.25, .03, 0.50, 0.02])
		samp = Slider(axamp, 'Time', 1, total_range[-1], valinit=0, valstep=1)

		rax = plt.axes([0.05, 0.7, 0.2, 0.25])
		eeg_visible = np.zeros(len(eeg_channels), dtype=bool)
		eeg_visible[0] = True
		check = CheckButtons(rax, eeg_channels, eeg_visible)

		is_manual = False
		interval = 0

		def eeg_check(label):
			eeg_lines[eeg_channels.index(label)].set_visible(not eeg_lines[eeg_channels.index(label)].get_visible())

		def update_slider(val):
			nonlocal is_manual
			is_manual = True
			val = int(val)
			update(val)

		def update(i):
			i = int(i + 1)
			
			if self.data["InterpGaze"] != None:
				line.set_xdata(self.data["InterpGaze"]["left"]["x"][:i])
				line.set_ydata(self.data["InterpGaze"]["left"]["y"][:i])
			
				circle.set_xdata(self.data["InterpGaze"]["left"]["x"][i])
				circle.set_ydata(self.data["InterpGaze"]["left"]["y"][i])
				
				line3.set_xdata(total_range[:i])
				line3.set_ydata(self.data["InterpPupilSize"][:i])
				ax3.set_ylim([min(self.data["InterpPupilSize"][:i]) - 5, max(self.data["InterpPupilSize"][:i]) + 5])      

			if self.data["EEG"] != None:
				for ind, channel in enumerate(viz_eeg):
					eeg_lines[ind].set_xdata(total_range[:i])
					eeg_lines[ind].set_ydata(viz_eeg[channel][:i])
					ax2.set_ylim([min(viz_eeg[channel][:i]) - 10, max(viz_eeg[channel][:i]) + 10])
		
			fig.canvas.draw_idle()

		def update_plot(i):
			nonlocal is_manual
			if is_manual:
				if self.data["EEG"] != None and self.data["InterpGaze"] != None:
					plots = [i for i in eeg_lines]
					plots.append(line)
					plots.append(line3)
					plots.append(circle)
					return plots
				elif self.data["EEG"] == None and self.data["InterpGaze"] != None:
					return [line, circle, line3]
				elif self.data["EEG"] != None and self.data["InterpGaze"] == None:
					return eeg_lines
				else:
					return

			i = int(samp.val + 1) % total_range[-1]
			samp.set_val(i)
			is_manual = False # the above line called update_slider, so we need to reset this
			fig.canvas.draw_idle()
			if self.data["EEG"] != None and self.data["InterpGaze"] != None:
				plots = [i for i in eeg_lines]
				plots.append(line)
				plots.append(line3)
				plots.append(circle)
				return plots
			elif self.data["EEG"] == None and self.data["InterpGaze"] != None:
				return [line, circle, line3]
			elif self.data["EEG"] != None and self.data["InterpGaze"] == None:
				return eeg_lines
			else:
				return

		def on_click(event):
			nonlocal is_manual
			# Check where the click happened
			(xm,ym),(xM,yM) = samp.label.clipbox.get_points()
			(xm2,ym2),(xM2, yM2) = check.label.clipbox.get_points()
			if (xm < event.x < xM and ym < event.y < yM) or (xm2 < event.x < xM2 and ym2 < event.y < yM2):
				# Event happened within the slider or checkbox, ignore since it is handled in update_slider
				return
			else:
				# user clicked somewhere else on canvas = unpause
				is_manual=False

		# call update function on slider value change
		samp.on_changed(update_slider)
		check.on_clicked(eeg_check)

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

		num_chars = 1
		if self.stim_type in ["alpha", "relevant", "general", "general_lie"]:
			with open("questions.json") as q_file:
				data = json.load(q_file)

			num_chars = len(data[self.name])

		# Finding response time based on number of  samples 
		self.response_time = len(self.data["ETRows"])

		# Find microsaccades
		all_MS, ms_count, ms_duration = self.findMicrosaccades(self.data["FixationSeq"], self.data["Gaze"])

		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["sacc_count"] = 0
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["sacc_duration"] = 0
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["ms_count"] = ms_count / self.response_time
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["ms_duration"] = ms_duration
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["pupil_size"] = self.data["InterpPupilSize"] - self.data["InterpPupilSize"][0]
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["blink_count"] = ((len(self.data["BlinksLeft"]["blink_onset"]) + len(self.data["BlinksRight"]["blink_onset"])) / 2) / self.response_time
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["fixation_count"] = (len(np.unique(self.data["FixationSeq"])) - 1) / num_chars
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["response_time"] = self.response_time / num_chars

		temp = np.empty(1, dtype='float32')
		for ms in all_MS:
			if ms["NB"] != 0:
				temp = np.hstack((temp, ms["bin"][:, 2]))
			if ms["NL"] != 0:
				temp = np.hstack((temp, ms["left"][:, 2]))
			if ms["NR"] != 0:
				temp = np.hstack((temp, ms["right"][:, 2]))
		if len(temp) != 0:
			self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["ms_vel"] = temp[1:]
		
		temp = np.empty(1, dtype='float32')
		for ms in all_MS:
			if ms["NB"] != 0:
				temp = np.hstack((temp, np.sqrt(ms["bin"][:, 5]**2 + ms["bin"][:, 6]**2)))
			if ms["NL"] != 0:
				temp = np.hstack((temp, np.sqrt(ms["left"][:, 5]**2 + ms["left"][:, 6]**2)))
			if ms["NR"] != 0:
				temp = np.hstack((temp, np.sqrt(ms["right"][:, 5]**2 + ms["right"][:, 6]**2)))
		
		if len(temp) != 0:
			self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["ms_amplitude"] = temp[1:]

		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["peak_pupil"] = max(self.data["InterpPupilSize"] - self.data["InterpPupilSize"][0])
		self.sensors[Sensor.sensor_names.index("EyeTracker")].metadata["time_to_peak_pupil"] = np.argmax(self.data["InterpPupilSize"] - self.data["InterpPupilSize"][0])


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
							"BlinksRight" : None,
							"EEG" : None,
							"EEGRows" : None}

		for col_class in sensor_names:
			if col_class in Sensor.sensor_names:
				self.sensors.append(Sensor(col_class))

			if col_class == "EyeTracker":
				event_type = np.array(data.EventSource)
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
				
				total_range = range(len(et_rows))

				# Extracting the eye gaze data
				et_rows = np.where(data.EventSource.str.contains("ET"))[0]
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
				blinks_l, interp_pupil_size_l, new_gaze_l = self.fix_blinks(pupil_size_l, gaze=gaze, interpolate=True, concat=True)
				blinks_r, interp_pupil_size_r, new_gaze_r = self.fix_blinks(pupil_size_r, gaze=gaze, interpolate=True, concat=True)
				interp_pupil_size = np.mean([interp_pupil_size_r, interp_pupil_size_l], axis=0)

				extracted_data["ETRows"] = et_rows
				extracted_data["FixationSeq"] = fixation_seq
				extracted_data["Gaze"] = gaze
				extracted_data["InterpPupilSize"] = interp_pupil_size
				extracted_data["InterpGaze"] = new_gaze_l
				extracted_data["BlinksLeft"] = blinks_l
				extracted_data["BlinksRight"] = blinks_r

			if col_class == "EEG":
				eeg_dict = {}
				for channel in contents["Columns_of_interest"][col_class]:
					eeg_df = np.array(data[channel])
					eeg_rows = np.where(data.EventSource.str.contains("Raw EEG Epoc"))[0]
					
					if len(eeg_rows) != 0:
						eeg = np.squeeze(np.array([eeg_df[i] for i in sorted(eeg_rows)], dtype="float32"))
						# (eeg_pz, eeg_time) = signal.resample(eeg_unique, len(total_range), t=sorted(eeg_rows))
					else:
						eeg = []

					channel_name = [i for i in Sensor.eeg_montage if i.upper() in channel.upper()][0]
					eeg_dict.update({channel_name:eeg})

				extracted_data["EEG"] = eeg_dict
				extracted_data["EEGRows"] = eeg_rows

		return extracted_data
