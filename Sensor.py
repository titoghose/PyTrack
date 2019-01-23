class Sensor:

	eye_cols = ["GazeX", "GazeY", "PupilLeft", "PupilRight", "FixationSeq"]
	eeg_cols = ["O1Pz_Epoc","SystemTimestamp_Epoc", "EmoTimestamp_Epoc"]

	sensor_names = ["Eye Tracker", "EEG"]
	meta_cols = [["sacc_count", "sacc_dur", "blink_count", "ms_count", "ms_duration", "pupil_size", "fixation_count"], []]

	def __init__(self, name):
		self.name = name 
		self.metadata = {key:None for key in self.meta_cols[self.sensor_names.index(name)]}
