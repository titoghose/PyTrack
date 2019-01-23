class Sensor:

	def __init__(self, name, col_names):
		self.name = name
		self.col_names = col_names
		self.metadata = {key:None for key in self.col_names}

s = Sensor("",["c", "d"])
print(s.metadata)
