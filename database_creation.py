import pandas as pd
import os
from sqlalchemy import create_engine
from datetime import datetime

na_strings = []
source_folder = "/media/arvind/My Passport/NTU Lie detection/Test data backup/csv files with column headers"


def null_value_return():

	for i in range(160):
		na_value = "Unnamed: " + str(i)
		na_strings.append(na_value)

	return na_strings	

def dtype_conversion_dict():

	dtype_dictionary = {"CameraLeftX" : "float"}

	return dtype_dictionary



def db_create(source_folder,database_name):

	na_strings = null_value_return()

	all_files = os.listdir(source_folder)

	newlist = []
	for names in all_files:
		if names.endswith(".csv"):
			newlist.append(names)

	database = "sqlite:///" + database_name + ".db"
	csv_database = create_engine(database)

	dtype_dictionary = dtype_conversion_dict()

	for file in newlist:

		print(file, datetime.now().time())

		file_name = source_folder + "/" + file

		length = len(file)
		file_name_no_extension = file[:length-4]
		table_name = file_name_no_extension.replace(' ','_') #To ensure table names dont haves spaces


		print(table_name)

		chunksize = 100000
		i = 0
		j = 1
		for df in pd.read_csv(file_name, chunksize=chunksize, iterator=True,na_values = na_strings,dtype= dtype_dictionary):
			
			df = df.rename(columns={c: c.replace(' ', '_') for c in df.columns})
			df = df.rename(columns={c: c.replace('/', '') for c in df.columns})
			df = df.rename(columns={c: c.replace('(', '') for c in df.columns})
			df = df.rename(columns={c: c.replace(')', '') for c in df.columns})

			#Explicit changing of 2 columns of data
			df.replace({"FixationAOI" : r'.Response.' }, {"FixationAOI": 2}, regex=True, inplace = True)
			df.replace({"FixationAOI" : r'.Q.' }, {"FixationAOI": 1}, regex=True, inplace = True)
			df.replace({"FixationAOI" : r'.Instructions.' }, {"FixationAOI": 1}, regex=True, inplace = True)
			df.replace({"FixationAOI" : r'nan' }, {"FixationAOI": -1}, regex=True, inplace = True)

			df.replace({"GazeAOI" : r'.Response.' }, {"GazeAOI": 2}, regex=True, inplace = True)
			df.replace({"GazeAOI" : r'.Instructions.' }, {"GazeAOI": 2}, regex=True, inplace = True)
			df.replace({"GazeAOI" : r'.Q.' }, {"GazeAOI": 1}, regex=True, inplace = True)
			df.replace({"GazeAOI" : r'nan' }, {"GazeAOI": -1}, regex=True, inplace = True)


			df.index += j
			i+=1
			df.to_sql(table_name, csv_database, if_exists='append',index = True, index_label = "Index")
			j = df.index[-1] + 1

#db_create(source_folder,"Lie_detection_database")
#print(datetime.now().time())

