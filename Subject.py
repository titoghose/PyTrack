class Subject:

	def __init__(self, name,subj_type,stimuli_names,columns,json_file):

		self.name = name
		self.subj_type = subj_type
		self.stimulus = self.stimulusDictInitialisation(stimuli_names,columns) #dictionary of objects of class stimulus demarcated by categories


	def dataExtraction(columns,json_file):

		'''
		Extracts the required columns from the data base and returns a pandas datastructure

		Input:
		1.	name_of_database: [string] name of the database
		2.	columns: [list] list of the names of the columns of interest

		Output:
		1.	df: [pandas datastructure] contains the data of columns of our interest
		'''

	    with open(json_file) as json_f:
			json_data = json.load(json_f)

		name_of_database = json_data[Database_name]

	    extended_name = "sqlite:///" + name_of_database
	    database = create_engine(extended_name)

	    string = 'SELECT '

	    index = 0

	    for name in columns:

	    	if index == 0:
	    		string = string + name
	    		index = index + 1

	    	else:   
	    		string = string + ',' + name
	    		index = index + 1

	    string = string + ' FROM "' + self.name + '"'

	    df = pd.read_sql_query(string, database)

	    return df
		

	def timeIndexInitialisation(stimulus_name):

		'''
		This function that will retireve the index of the start, end and roi of a question

		Input:
		1.	stimulus_name: [string] Name of the stimulus 

		Output:
		1.	start:[integer] the index of the start of a queation
		2.	end:[integer] the index of the end of a question
		3.	roi:[integer] the index when the eye lands on the region of interest
		'''

		index = df[df[question_column_name] == stimulus_name].index

        try:
            start = min(index)
            end = max(index)
            roi = -1
        except:
            print("Error: ", question_name)
            start = -1
            end = -1
            roi = -1

        return start,end,roi

	
	def stimulusDictInitialisation(stimuli_names,columns):

		'''
		Creates  a list of objects of class Stimuli

		Input:
		1.	stimuli_names:[list] list of names of different stimulus
		2.	columns: [list] list of names of the columns of interest

		Output:
		1.	stimulus_object_dict: [dictionary] dictionary of objects of class stimulus ordered by category
		'''	

		question_column = dataExtraction(self.name,["StimulusName"])

		data = dataExtraction(self.name,columns)

		stimulus_object_dict = {}

		for category in stimuli_names:
			
			stimulus_object_list = []

			for name in stimuli_names[category]:

				#Check if the pickle file exists, if it exists then pull data from it else get it from the 


				start_time,end_time,roi_time = self.timeIndexInitialisation(stimulus_name)
				
				stimuli_data = data[start_time:end_time+1]

				stimulus_object = stimulus(name, category, ["Eye Tracker", "EEG"], stimuli_data, start_time, end_time, roi_time)

				stimulus_object_list.append(stimulus_object)

			stimulus_object_dict[k] = stimulus_object_list

		return stimulus_object_dict