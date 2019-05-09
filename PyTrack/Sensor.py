# -*- coding: utf-8 -*-

class Sensor:
	"""This class represents the paramters of the sensors used during the experiment.

	As of now the sensor class only support the Eye Tracker but in future versions, we plan to include EEG, ECG, EDA and Respiration as well. The class is used to store all the metadata/features extracted during analysis.

	Attributes
	----------
	sensor_names : list(str)
		List of accepted sensors.
	meta_cols : dict
		Dictionary of lists containing the various metadata/features of a given sensor.

	Parameters
	----------
	name : str
		Name of the sensor.
	sampling_freq : int | float
		Sampling frequency of the sensor.

	"""

	sensor_names = ["EyeTracker"]
	meta_cols = {"EyeTracker": ["response_time", "pupil_size", "time_to_peak_pupil", "peak_pupil", "pupil_mean", "pupil_size_downsample", "pupil_slope", "pupil_area_curve", "blink_rate", "peak_blink_duration", "avg_blink_duration", "fixation_count", "max_fixation_duration", "avg_fixation_duration", "sacc_count", "sacc_duration", "sacc_vel", "sacc_amplitude", "ms_count", "ms_duration", "ms_vel", "ms_amplitude", "no_revisits", "first_pass", "second_pass"]}

	def __init__(self, name, sampling_freq):
		self.name = name
		self.metadata = {key:None for key in self.meta_cols[name]}
		self.sampling_freq = sampling_freq


