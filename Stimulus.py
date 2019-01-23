from Sensor import Sensor

class Stimulus:

	def __init__(self, name, stim_type, sensor_names, data, start_time, end_time, roi_time):
		self.name = name
		self.stim_type = stim_type
		self.data = data
		self.start_time = start_time
		self.end_time = end_time
		self.roi_time = roi_time
		self.sensors = []
		for sn in sensor_names:
			self.sensors.append(Sensor(sn))


	def diff(self, series):
		"""
		Python implementation of Matlab's 'diff' function. Returns (a[n+1] - a[n]) for all n.
		"""
		return series[1:] - series[:-1]


	def smooth(self, x, window_len=10, window='flat'):
		



		"""
		SOURCE: https://scipy-cookbook.readthedocs.io/index.html
		
		smooth the data using a window with requested size.
		
		This method is based on the convolution of a scaled window with the signal.
		The signal is prepared by introducing reflected copies of the signal 
		(with the window size) in both ends so that transient parts are minimized
		in the begining and end part of the output signal.
		
		input:
			x: the input signal 
			window_len: the dimension of the smoothing window; should be an odd integer
			window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
				flat window will produce a moving average smoothing.
		
		output:
			the smoothed signal
			
		see also: 
		
		numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
		scipy.signal.lfilter
		
		TODO: the window parameter could be the window itself if an array instead of a string
		NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
		"""
		
		if window_len < 3:
			return x
		
		s = np.r_[x[window_len - 1: 0: -1], x, x[-2: -window_len - 1: -1]]
		# print(len(s))
		if window == 'flat':  # moving average
			w = np.ones(window_len, 'd')
		else:
			w = eval('numpy.' + window + '(window_len)')
		
		y = np.convolve(w / w.sum(), s, mode='valid')
		return y[(window_len // 2 - 1):-(window_len // 2)]


	def fix_blinks(self, pupil_size, gaze, sampling_freq=1000, concat=False, concat_gap_interval = 100):
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
		difference = diff(missing_data)
		
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
			blink_onset.append(len(difference))
		
		ms_4_smoothing = 10
		samples2smooth = ms_4_smoothing // sampling_interval
		smooth_pupil_size = smooth(pupil_size, 10, "flat")
		
		smooth_pupil_size[np.where(pupil_size == -1)[0]] = float('nan')
		smooth_pupil_size_diff = diff(smooth_pupil_size)
		
		monotonically_dec = smooth_pupil_size_diff <= 0
		monotonically_inc = smooth_pupil_size_diff >= 0
		
		for i in range(len(blink_onset)):
			j = blink_onset[i] - 1
			while j > 0 and monotonically_dec[j] == True:
				j -= 1
			blink_onset[i] = j
		
			j = blink_offset[i]
			while j < len(monotonically_inc) and monotonically_inc[j] == True:
				j += 1
			blink_offset[i] = j
		
		if concat:
			for i in range(len(blink_onset) - 1):
				if blink_onset[i + 1] - blink_offset[i] <= concat_gap_interval:
					blink_onset[i + 1] = blink_onset[i]
					blink_offset[i] = blink_offset[i + 1]
		
			blink_onset = np.unique(blink_onset)
			blink_offset = np.unique(blink_offset)
		
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
		
		gaze_x = gaze["x"]
		gaze_y = gaze["y"]
		
		gaze_x_no_blinks = gaze_x[pupil_size_no_blinks_indices]
		gaze_y_no_blinks = gaze_y[pupil_size_no_blinks_indices]
		
		interp_gaze_x = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices), gaze_x_no_blinks)
		interp_gaze_y = np.interp(np.arange(len(pupil_size)), sorted(pupil_size_no_blinks_indices), gaze_y_no_blinks)
		
		new_gaze = {"x": interp_gaze_x, "y": interp_gaze_y}
		
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
		fixation_ind_diff = diff(fixation_ind)
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
		v1 = vel[1] + smooth_gaze[0]
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


	def findMicrosaccades(self, fixation_seq, gaze, sampling_freq=1000):
		"""
		Function to detect microsaccades within fixations.
		Adapted from R. Engbert and K. Mergenthaler, “Microsaccades are triggered by low retinal image slip,” Proc. Natl. Acad. Sci., vol. 103, no. 18, pp. 7192–7197, 2006.
		
		Input:
		
		Output:
		
		"""
		MINDUR = (1000/sampling_freq) * 6
		gaze_x = gaze["x"]
		gaze_y = gaze["y"]
		fixation_indices = findFixationsIDT(fixation_seq)
		
		all_MS = []
		
		for i in range(len(fixation_indices["start"])):
		
			curr_gaze_x = gaze_x[fixation_indices["start"][i] : fixation_indices["end"][i]]
			curr_gaze_y = gaze_y[fixation_indices["start"][i] : fixation_indices["end"][i]]
		
			vel_x = position2Velocity(curr_gaze_x, sampling_freq)
			vel_y = position2Velocity(curr_gaze_y, sampling_freq)
		
			smooth_gaze_x = smoothGaze(vel_x, curr_gaze_x, sampling_freq)
			smooth_gaze_y = smoothGaze(vel_y, curr_gaze_y, sampling_freq)	
		
			radius_x = calculateMSThreshold(vel_x, sampling_freq)
			radius_y = calculateMSThreshold(vel_y, sampling_freq)
		
			temp = (vel_x/radius_x)**2 + (vel_y/radius_y)**2
			ms_indices = np.where(temp > 1)[0]
		
			N = len(ms_indices) 
			num_ms = 0
			MS = np.zeros((1, 7))
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
							new_ms = np.array([ms_indices[a], ms_indices[b], 0, 0, 0, 0, 0])
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
					new_ms = np.array([ms_indices[a], ms_indices[b], 0, 0, 0, 0, 0])
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
					dx = curr_gaze_x[b] - curr_gaze_x[a]
					dy = curr_gaze_y[b] - curr_gaze_y[a]
					MS[s][3] = dx
					MS[s][4] = dy
		
					# Saccade amplitude (dX,dY)
					minx = min(curr_gaze_x[idx])
					maxx = max(curr_gaze_x[idx])
					miny = min(curr_gaze_y[idx])
					maxy = max(curr_gaze_y[idx])
					ix1 = np.argmin(curr_gaze_x[idx])
					ix2 = np.argmax(curr_gaze_x[idx])
					iy1 = np.argmin(curr_gaze_y[idx])
					iy2 = np.argmax(curr_gaze_y[idx])
					dX = np.sign(ix2 - ix1) * (maxx - minx)
					dY = np.sign(iy2 - iy1) * (maxy - miny)
					MS[s][5] = dX
					MS[s][6] = dY
		
					all_MS.append((MS[s], radius_x, radius_y))
		
				# fig = plt.figure()
				
				# a1 = fig.add_subplot(2, 1, 1)
				# a1.plot(smooth_gaze_x, smooth_gaze_y)
				# for i in range(len(MS)):
				# 	a1.plot(smooth_gaze_x[int(MS[i][0]) : int(MS[i][1])], smooth_gaze_y[int(MS[i][0]) : int(MS[i][1])], color='r')
		
				
				# a2 = fig.add_subplot(2, 1, 2)
				# e = Ellipse((0, 0), radius_x, radius_y, linestyle='--', color='g', fill=False)
				# a2.add_artist(e)
		
				# a2.plot(vel_x, vel_y, alpha=0.5)
				# for i in range(len(MS)):
				# 	a2.plot(vel_x[int(MS[i][0]) : int(MS[i][1])], vel_y[int(MS[i][0]) : int(MS[i][1])], color='r') 
		
				# plt.show()
		
		ms_count = 0
		ms_duration = []
		
		ms_count = len(all_MS)
		for ms in all_MS:
			ms_duration.append(ms[0][1] - ms[0][0])
		return all_MS, ms_count, ms_duration


	def findEyeMetaData(self):
		"""
		Input:
			subject_name : [string] Name of subject to visualize data for 
			stimuli_name : [list] of strings containing stimuli names to visualize the data for
		Output:
			NA
		"""

		# Extracting data for particular stimulus
		event_type = self.data.EventType
		gazex_df = self.data.GazeX
		gazey_df = self.data.GazeY
		eeg_pz_df = self.data.O1Pz_Epoc
		pupil_size_l_df = self.data.PupilLeft
		pupil_size_r_df = self.data.PupilRight
		fixation_seq_df = self.data.FixationSeq
		
		# Extracting fixation sequences
		et_rows = np.where(event_type.EventSource.str.contains("ET"))[0]
		fixation_seq_df = np.array(fixation_seq_df.FixationSeq.fillna(-1), dtype='float32')
		fixation_seq = np.squeeze(np.array([fixation_seq_df[i] for i in sorted(et_rows)], dtype="float32"))
		
		# Extracting the eye gaze data
		et_rows = np.where(event_type.EventSource.str.contains("ET"))[0]
		gaze_x = np.squeeze(np.array([gazex_df.GazeX[i] for i in sorted(et_rows)], dtype="float32"))
		gaze_y = np.squeeze(np.array([gazey_df.GazeY[i] for i in sorted(et_rows)], dtype="float32"))
		gaze = {"x": gaze_x, "y": gaze_y}
		
		# Extracting Pupil Size Data
		pupil_size_r = np.squeeze(np.array([pupil_size_r_df.PupilRight[i] for i in sorted(et_rows)], dtype="float32"))
		pupil_size_l = np.squeeze(np.array([pupil_size_l_df.PupilLeft[i] for i in sorted(et_rows)], dtype="float32"))
		pupil_size = np.mean([pupil_size_r, pupil_size_l], axis=0)
		
		# Fixing Blinks and interpolating pupil size and gaze data
		blinks, interp_pupil_size, new_gaze = fix_blinks(pupil_size, gaze)
		gaze_x, gaze_y = None, None
		gaze_x = new_gaze["x"]
		gaze_y = new_gaze["y"]

		# Find microsaccades
		all_MS, ms_count, ms_duration = findMicrosaccades(fixation_seq, gaze)

		sensors[Sensor.sensor_names.index("Eye Tracker")].metadata["ms_count"] = ms_count
		sensors[Sensor.sensor_names.index("Eye Tracker")].metadata["ms_duration"] = ms_duration
		sensors[Sensor.sensor_names.index("Eye Tracker")].metadata["pupil_size"] = interp_pupil_size
		sensors[Sensor.sensor_names.index("Eye Tracker")].metadata["blink_count"] = len(blinks)
		sensors[Sensor.sensor_names.index("Eye Tracker")].metadata["fixation_count"] = len(np.unique(fixation_seq)) - 1

