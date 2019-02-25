class Sensor:

	sensor_names = ["Eye Tracker", "EEG"]
	meta_cols = [["time_to_peak_pupil", "peak_pupil", "sacc_count", "sacc_duration", "blink_count", "ms_count" , "ms_duration", "ms_vel", "ms_amplitude", "pupil_size", "fixation_count", "response_time"], []]

	eeg_montage = []

	def __init__(self, name):
		global eeg_montage
		self.name = name 
		self.metadata = {key:None for key in self.meta_cols[self.sensor_names.index(name)]}

		with open("1020_montage.txt") as f:
			content = f.readlines()

		eeg_montage = [line.strip() for line in content]


