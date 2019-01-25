class Subject:

	def __init__(self, name,subj_type,stimuli_names,columns):

		self.name = name
		self.subj_type = subj_type
		self.stimulus = stimulusDictInitialisation(stimuli_names,columns) #dictionary of objects of class stimulus demarcated by categories


	def dataExtraction(name_of_database,columns):

		'''
		Extracts the required columns from the data base and returns a pandas datastructure

		Input:

		name_of_database: [string] name of the database
		columns: [list] list of the names of the columns of interest

		Output:

		df: [pandas datastructure] contains the data of columns of our interest
		'''

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

		stimulus_name: [string] 

		Output:

		start:[integer] the index of the start of a queation

		end:[integer] the index of the end of a question

		roi:[integer] the index when the eye lands on the region of interest
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

	stimuli_names:[list] list of names of different stimulus

	columns: [list] list of names of the columns of interest

	Output:

	stimulus_object_dict: [dictionary] dictionary of objects of class stimulus ordered by category
	'''	

		question_column = dataExtraction(name_of_database,self.name,["StimulusName"])

		data = dataExtraction(name_of_database,self.name,columns)


		stimulus_object_dict = {}

		for category in stimuli_names:
			
			stimulus_object_list = []

			for name in stimuli_names[category]:

				start_time,end_time,roi_time = self.timeIndexInitialisation(stimulus_name)

				stimuli_data = data[start:end+1]

				stimulus_object = stimulus(name, category, ["Eye Tracker", "EEG"], stimuli_data, start_time, end_time, roi_time)

				stimulus_object_list.append(stimulus_object)

			stimulus_object_dict[k] = stimulus_object_list


		return stimulus_object_dict