class Stimulus:

	def __init__(self, name, stim_type, sensors, data):
		self.name = name
		self.stim_type = stim_type
		self.data = data
		self.start_time = 0
		self.end_time = 0
		self.roi_time = 0
		self.sensors = None
		
s = Stimulus("dn", "as", ["adc"], []) 
