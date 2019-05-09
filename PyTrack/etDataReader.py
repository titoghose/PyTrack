# -*- coding: utf-8 -*-

import copy
import os.path
import numpy

def blink_detection(x, y, time, missing=0.0, minlen=10):
	"""Detects blinks, defined as a period of missing data that lasts for at least a minimal amount of samples

	Parameters
	----------
	x : array
		Gaze x positions
	y :	array
		Gaze y positions
	time : array
		Timestamps
	missing	: float
		Value to be used for missing data (default = 0.0)
	minlen : int
		Minimal amount of consecutive missing samples

	Returns
	-------
	Sblk : list of lists
		Each containing [starttime]
	Eblk : list of lists
		Each containing [starttime, endtime, duration]

	"""

	# empty list to contain data
	Sblk = []
	Eblk = []

	# check where the missing samples are
	mx = numpy.array(x==missing, dtype=int)
	my = numpy.array(y==missing, dtype=int)
	miss = numpy.array((mx+my) == 2, dtype=int)

	# check where the starts and ends are (+1 to counteract shift to left)
	diff = numpy.diff(miss)
	starts = numpy.where(diff==1)[0] + 1
	ends = numpy.where(diff==-1)[0] + 1

	# compile blink starts and ends
	for i in range(len(starts)):
		# get starting index
		s = starts[i]
		# get ending index
		if i < len(ends):
			e = ends[i]
		elif len(ends) > 0:
			e = ends[-1]
		else:
			e = -1
		# append only if the duration in samples is equal to or greater than
		# the minimal duration
		if e-s >= minlen:
			# add starting time
			Sblk.append([time[s]])
			# add ending time
			Eblk.append([time[s],time[e],time[e]-time[s]])

	return Sblk, Eblk


def fixation_detection(x, y, time, missing=0.0, maxdist=25, mindur=50):
	"""Detects fixations, defined as consecutive samples with an inter-sample distance of less than a set amount of pixels (disregarding missing data)

	Parameters
	----------
	x : array
		Gaze x positions
	y :	array
		Gaze y positions
	time : array
		Timestamps
	missing	: float
		Value to be used for missing data (default = 0.0)
	maxdist : int
		Maximal inter sample distance in pixels (default = 25)
	mindur : int
		Minimal duration of a fixation in milliseconds; detected fixation cadidates will be disregarded if they are below this duration (default = 100)

	Returns
	-------
	Sfix : list of lists
		Each containing [starttime]
	Efix : list of lists
		Each containing [starttime, endtime, duration, endx, endy]

	"""

	# empty list to contain data
	Sfix = []
	Efix = []

	# loop through all coordinates
	si = 0
	fixstart = False
	for i in range(1,len(x)):
		# calculate Euclidean distance from the current fixation coordinate
		# to the next coordinate
		dist = ((x[si]-x[i])**2 + (y[si]-y[i])**2)**0.5
		# check if the next coordinate is below maximal distance
		if dist <= maxdist and not fixstart:
			# start a new fixation
			si = 0 + i
			fixstart = True
			Sfix.append([time[i]])
		elif dist > maxdist and fixstart:
			# end the current fixation
			fixstart = False
			# only store the fixation if the duration is ok
			if time[i-1]-Sfix[-1][0] >= mindur:
				Efix.append([Sfix[-1][0], time[i-1], time[i-1]-Sfix[-1][0], x[si], y[si]])
			# delete the last fixation start if it was too short
			else:
				Sfix.pop(-1)
			si = 0 + i
		elif not fixstart:
			si += 1

	return Sfix, Efix


def saccade_detection(x, y, time, missing=0.0, minlen=5, maxvel=40, maxacc=340):
	"""Detects saccades, defined as consecutive samples with an inter-sample velocity of over a velocity threshold or an acceleration threshold

	Parameters
	----------
	x : array
		Gaze x positions
	y :	array
		Gaze y positions
	time : array
		Timestamps
	missing	: float
		Value to be used for missing data (default = 0.0)
	minlen : int
		Minimal length of saccades in milliseconds; all detected saccades with len(sac) < minlen will be ignored (default = 5)
	maxvel : int
		Velocity threshold in pixels/second (default = 40)
	maxacc : int
		Acceleration threshold in pixels / second**2 (default = 340)

	Returns
	-------
	Ssac : list of lists
		Each containing [starttime]
	Esac : list of lists
		Each containing [starttime, endtime, duration, startx, starty, endx, endy]

	"""

	# CONTAINERS
	Ssac = []
	Esac = []

	# INTER-SAMPLE MEASURES
	# the distance between samples is the square root of the sum
	# of the squared horizontal and vertical interdistances
	intdist = (numpy.diff(x)**2 + numpy.diff(y)**2)**0.5
	# get inter-sample times
	inttime = numpy.diff(time)
	# recalculate inter-sample times to seconds
	inttime = inttime / 1000.0

	# VELOCITY AND ACCELERATION
	# the velocity between samples is the inter-sample distance
	# divided by the inter-sample time
	vel = intdist / inttime
	# the acceleration is the sample-to-sample difference in
	# eye movement velocity
	acc = numpy.diff(vel)

	# SACCADE START AND END
	t0i = 0
	stop = False
	while not stop:
		# saccade start (t1) is when the velocity or acceleration
		# surpass threshold, saccade end (t2) is when both return
		# under threshold

		# detect saccade starts
		sacstarts = numpy.where((vel[1+t0i:] > maxvel).astype(int) + (acc[t0i:] > maxacc).astype(int) >= 1)[0]
		if len(sacstarts) > 0:
			# timestamp for starting position
			t1i = t0i + sacstarts[0] + 1
			if t1i >= len(time)-1:
				t1i = len(time)-2
			t1 = time[t1i]

			# add to saccade starts
			Ssac.append([t1])

			# detect saccade endings
			sacends = numpy.where((vel[1+t1i:] < maxvel).astype(int) + (acc[t1i:] < maxacc).astype(int) == 2)[0]
			if len(sacends) > 0:
				# timestamp for ending position
				t2i = sacends[0] + 1 + t1i + 2
				if t2i >= len(time):
					t2i = len(time)-1
				t2 = time[t2i]
				dur = t2 - t1

				# ignore saccades that did not last long enough
				if dur >= minlen:
					# add to saccade ends
					Esac.append([t1, t2, dur, x[t1i], y[t1i], x[t2i], y[t2i]])
				else:
					# remove last saccade start on too low duration
					Ssac.pop(-1)

				# update t0i
				t0i = 0 + t2i
			else:
				stop = True
		else:
			stop = True

	return Ssac, Esac


def read_idf(filename, start, stop=None, missing=0.0, debug=False):
	"""Returns a list with dicts for every trial.

	Parameters
	----------
	filename : str
		Path to the file that has to be read
	start : str
		Trial start string
	stop : str
		Trial ending string (default = None)
	missing : float
		Value to be used for missing data (default = 0.0)
	debug : bool
		Indicating if DEBUG mode should be on or off; if DEBUG mode is on, information on what the script currently is doing will be printed to the console (default = False)

	Returns
	-------
	data : list
		With a dict for every trial. Following is the dictionary
		0. x -array of Gaze x positions,
		1. y -array of Gaze y positions,
		2. size -array of pupil size,
		3. time -array of timestamps, t=0 at trialstart,
		4. trackertime -array of timestamps, according to the tracker,
		5. events -dict {Sfix, Ssac, Sblk, Efix, Esac, Eblk, msg}

	"""

	# # # # #
	# debug mode

	if debug:
		def message(msg):
			print(msg)
	else:
		def message(msg):
			pass


	# # # # #
	# file handling

	# check if the file exists
	if os.path.isfile(filename):
		# open file
		message("opening file '%s'" % filename)
		f = open(filename, 'r')
	# raise exception if the file does not exist
	else:
		raise Exception("Error in read_idf: file '%s' does not exist" % filename)

	# read file contents
	message("reading file '%s'" % filename)
	raw = f.readlines()

	# close file
	message("closing file '%s'" % filename)
	f.close()


	# # # # #
	# parse lines

	# variables
	data = []
	x_l = []
	y_l = []
	x_r = []
	y_r = []
	size_l = []
	size_r = []
	time = []
	trackertime = []
	events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
	starttime = 0
	started = False
	trialend = False
	filestarted = False

	timei = None
	typei = None
	msgi = -1
	xi = {'L':None, 'R':None}
	yi = {'L':None, 'R':None}
	sizei = {'L':None, 'R':None}

	# loop through all lines
	for i in range(len(raw)):

		# string to list
		line = raw[i].replace('\n','').replace('\r','').split('\t')
		# check if the line starts with '##' (denoting header)
		if '##' in line[0]:
			# skip processing
			continue
		elif '##' not in line[0]	 and not filestarted:
			# check the indexes for several key things we want to extract
			# (we need to do this, because ASCII outputs of the IDF reader
			# are different, based on whatever the user wanted to extract)

			timei = line.index("Time")
			typei = line.index("Type")
			msgi = -1
			xi = {'L':None, 'R':None}
			yi = {'L':None, 'R':None}
			sizei = {'L':None, 'R':None}
			if "L POR X [px]" in line:
				xi['L']  = line.index("L POR X [px]")
			if "R POR X [px]" in line:
				xi['R']  = line.index("R POR X [px]")
			if "L POR Y [px]" in line:
				yi['L']  = line.index("L POR Y [px]")
			if "R POR Y [px]" in line:
				yi['R']  = line.index("R POR Y [px]")
			if "L Dia [mm]" in line:
				sizei['L']  = line.index("L Dia [mm]")
			if "R Dia [mm]" in line:
				sizei['R']  = line.index("R Dia [mm]")
			# set filestarted to True, so we don't attempt to extract
			# this info on all consecutive lines
			filestarted = True

		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop != None:
				if stop in line[msgi] or i == len(raw)-1:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if start in line or i == len(raw)-1:
					started = True
					trialend = True

			# # # # #
			# trial ending

			if trialend:
				message("trialend %d; %d samples found" % (len(data),len(x_l)))
				# message("trialend %d; %d x_r samples found" % (len(data),len(x_r)))
				# message("trialend %d; %d y_l samples found" % (len(data),len(y_l)))
				# message("trialend %d; %d y_r samples found" % (len(data),len(y_r)))
				# message("trialend %d; %d size_l samples found" % (len(data),len(size_l)))
				# message("trialend %d; %d size_r samples found" % (len(data),len(size_r)))
				# trial dict
				trial = {}
				trial['x_l'] = numpy.array(x_l)
				trial['y_l'] = numpy.array(y_l)
				trial['x_r'] = numpy.array(x_r)
				trial['y_r'] = numpy.array(y_r)
				trial['size_l'] = numpy.array(size_l)
				trial['size_r'] = numpy.array(size_r)
				trial['time'] = numpy.array(time)
				trial['trackertime'] = numpy.array(trackertime)
				trial['events'] = copy.deepcopy(events)
				# events
				trial['events']['Sblk'], trial['events']['Eblk'] = blink_detection(trial['x_l'],trial['y_l'],trial['trackertime'],missing=missing)
				trial['events']['Sfix'], trial['events']['Efix'] = fixation_detection(trial['x_l'],trial['y_l'],trial['trackertime'],missing=missing)
				trial['events']['Ssac'], trial['events']['Esac'] = saccade_detection(trial['x_l'],trial['y_l'],trial['trackertime'],missing=missing)
				# add trial to data
				data.append(trial)
				# reset stuff
				x_l = []
				y_l = []
				x_r = []
				y_r = []
				size_l = []
				size_r = []
				time = []
				trackertime = []
				events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
				trialend = False

		# check if the current line contains start message
		else:
			if start in line[msgi]:
				message("trialstart %d" % len(data))
				# set started to True
				started = True
				# find starting time
				starttime = int(line[timei])

		# # # # #
		# parse line

		if started:
			# message lines will usually start with a timestamp, followed
			# by 'MSG', the trial number and the actual message, e.g.:
			#	"7818328012	MSG	1	# Message: 3"
			if line[typei] == "MSG":
				t = int(line[timei]) # time
				m = line[msgi] # message
				events['msg'].append([t,m])

			# regular lines will contain tab separated values, beginning with
			# a timestamp, follwed by the values that were chosen to be
			# extracted by the IDF converter
			else:
				# see if current line contains relevant data
				try:
					# extract data on POR and pupil size
					vi = None
					for ind, var in enumerate([xi, yi, sizei]):
						vi = var
						val_l = -1
						val_r = -1
						# nothing
						if vi['L'] == None and vi['R'] == None:
							continue
						# only left eye
						elif vi['L'] != None and vi['R'] == None:
							if float(line[vi['L']]) == 0:
								line[vi['L']] = str(missing)
							val_l = float(line[vi['L']])
							val_r = val_l
						# only right eye
						elif vi['L'] == None and vi['R'] != None:
							if float(line[vi['R']]) == 0:
								line[vi['R']] = str(missing)
							val_r = float(line[vi['R']])
							val_l = val_r
						# average the two eyes, but only if they both
						# contain valid data
						elif vi['L'] != None and vi['R'] != None:
							if float(line[vi['R']]) == 0:
								line[vi['R']] = str(missing)
							if float(line[vi['L']]) == 0:
								line[vi['L']] = str(missing)
							val_r = float(line[vi['R']])
							val_l = float(line[vi['L']])

						if ind == 0:
							x_l.append(val_l)
							x_r.append(val_r)
						elif ind == 1:
							y_l.append(val_l)
							y_r.append(val_r)
						elif ind == 2:
							size_l.append(val_l)
							size_r.append(val_r)

					# extract time data
					time.append(int(line[timei])-starttime)
					trackertime.append(int(line[timei]))

				except:
					message("line '%s' could not be parsed" % line)
					continue # skip this line

	# # # # #
	# return

	return data


def replace_missing(value, missing=0.0):
	"""Returns missing code if passed value is missing, or the passed value if it is not missing; a missing value in the EDF contains only a period, no numbers; NOTE: this function is for gaze position values only, NOT for pupil size, as missing pupil size data is coded '0.0'

	Parameters
	----------
	value : str
		Either an X or a Y gaze position value (NOT pupil size! This is coded '0.0')
	missing : float
		The missing code to replace missing data with (default = 0.0)

	Returns
	-------
	float
		Either a missing code, or a float value of the gaze position

	"""

	if value.replace(' ','') == '.':
		return missing
	else:
		return float(value)


def read_edf(filename, start, stop=None, missing=0.0, debug=False, eye="B"):
	"""Returns a list with dicts for every trial.

	Parameters
	----------
	filename : str
		Path to the file that has to be read
	start : str
		Trial start string
	stop : str
		Trial ending string (default = None)
	missing : float
		Value to be used for missing data (default = 0.0)
	debug : bool
		Indicating if DEBUG mode should be on or off; if DEBUG mode is on, information on what the script currently is doing will be printed to the console (default = False)
	eye : str {'B','L','R'}
		Which eye is being tracked? Deafults to 'B'-Both. ['L'-Left, 'R'-Right, 'B'-Both]

	Returns
	-------
	data : list
		With a dict for every trial. Following is the dictionary
		0. x -array of Gaze x positions,
		1. y -array of Gaze y positions,
		2. size -array of pupil size,
		3. time -array of timestamps, t=0 at trialstart,
		4. trackertime -array of timestamps, according to the tracker,
		5. events -dict {Sfix, Ssac, Sblk, Efix, Esac, Eblk, msg}

	"""

	# # # # #
	# debug mode

	if debug:
		def message(msg):
			print(msg)
	else:
		def message(msg):
			pass


	# # # # #
	# file handling

	# check if the file exists
	if os.path.isfile(filename):
		# open file
		message("opening file '%s'" % filename)
		f = open(filename, 'r')
	# raise exception if the file does not exist
	else:
		raise Exception("Error in read_edf: file '%s' does not exist" % filename)

	# read file contents
	message("reading file '%s'" % filename)
	raw = f.readlines()

	# close file
	message("closing file '%s'" % filename)
	f.close()


	# # # # #
	# parse lines

	# variables
	data = []
	x_l = []
	y_l = []
	x_r = []
	y_r = []
	size_l = []
	size_r = []
	time = []
	trackertime = []
	events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
	starttime = 0
	started = False
	trialend = False
	finalline = raw[-1]
	fixation_flag = 0
	saccade_flag = 0
	blink_flag = 0

	which_eye = ''
	if eye == 'B':
		which_eye = 'L'
	else:
		which_eye = eye

	# loop through all lines
	for line in raw:

		# check if trial has already started
		if started:
			# only check for stop if there is one
			if stop != None:
				if stop in line:
					started = False
					trialend = True
			# check for new start otherwise
			else:
				if (start in line) or (line == finalline):
					started = True
					trialend = True
					fixation_flag = 0
					saccade_flag = 0
					blink_flag = 0

			# # # # #
			# trial ending

			if trialend:
				message("trialend %d; %d samples found" % (len(data),len(x_l)))
				# trial dict
				trial = {}
				trial['x_l'] = numpy.array(x_l)
				trial['y_l'] = numpy.array(y_l)
				trial['x_r'] = numpy.array(x_r)
				trial['y_r'] = numpy.array(y_r)
				trial['size_l'] = numpy.array(size_l)
				trial['size_r'] = numpy.array(size_r)
				trial['time'] = numpy.array(time)
				trial['trackertime'] = numpy.array(trackertime)
				trial['events'] = copy.deepcopy(events)
				# add trial to data
				data.append(trial)
				# reset stuff
				x_l = []
				y_l = []
				size_l = []
				x_r = []
				y_r = []
				size_r = []
				time = []
				trackertime = []
				events = {'Sfix':[],'Ssac':[],'Sblk':[],'Efix':[],'Esac':[],'Eblk':[],'msg':[]}
				trialend = False

		# check if the current line contains start message
		else:
			if start in line:
				message("trialstart %d" % len(data))
				# set started to True
				started = True
				# find starting time
				starttime = int(line[line.find('\t')+1:line.find(' ')])
				fixation_flag = 0
				saccade_flag = 0
				blink_flag = 0

		# # # # #
		# parse line

		if started:
			# message lines will start with MSG, followed by a tab, then a
			# timestamp, a space, and finally the message, e.g.:
			#	"MSG\t12345 something of importance here"
			if line[0:3] == "MSG":
				ms = line.find(" ") # message start
				t = int(line[4:ms]) # time
				m = line[ms+1:] # message
				events['msg'].append([t,m])

			# EDF event lines are constructed of 9 characters, followed by
			# tab separated values; these values MAY CONTAIN SPACES, but
			# these spaces are ignored by float() (thank you Python!)

			# fixation start
			elif line[0:6] == ("SFIX " + which_eye):
				message("fixation start")
				l = line[9:]
				if len(events['Sfix']) > len(events['Efix']):
					events['Sfix'][-1] = int(l)
				else:
					events['Sfix'].append(int(l))
				fixation_flag = 1

			# fixation end
			elif line[0:6] == ("EFIX " + which_eye) and fixation_flag:
				message("fixation end")
				l = line[9:]
				l = l.split('\t')
				st = int(l[0]) # starting time
				et = int(l[1]) # ending time
				dur = int(l[2]) # duration
				sx = replace_missing(l[3], missing=missing) # x position
				sy = replace_missing(l[4], missing=missing) # y position
				events['Efix'].append([st, et, dur, sx, sy])
				fixation_flag = 0

			# saccade start
			elif line[0:7] == ("SSACC " + which_eye):
				message("saccade start")
				l = line[9:]
				if len(events['Ssac']) > len(events['Esac']):
					events['Ssac'][-1] = int(l)
				else:
					events['Ssac'].append(int(l))
				saccade_flag = 1

			# saccade end
			elif line[0:7] == ("ESACC " + which_eye) and saccade_flag:
				message("saccade end")
				l = line[9:]
				l = l.split('\t')
				st = int(l[0]) # starting time
				et = int(l[1]) # endint time
				dur = int(l[2]) # duration
				sx = replace_missing(l[3], missing=missing) # start x position
				sy = replace_missing(l[4], missing=missing) # start y position
				ex = replace_missing(l[5], missing=missing) # end x position
				ey = replace_missing(l[6], missing=missing) # end y position
				events['Esac'].append([st, et, dur, sx, sy, ex, ey])
				saccade_flag = 0

			# blink start
			elif line[0:8] == ("SBLINK " + which_eye):
				message("blink start")
				l = line[9:]
				if len(events['Sblk']) > len(events['Eblk']):
					events['Sblk'][-1] = int(l)
				else:
					events['Sblk'].append(int(l))
				blink_flag = 1

			# blink end
			elif line[0:8] == ("EBLINK " + which_eye) and blink_flag:
				message("blink end")
				l = line[9:]
				l = l.split('\t')
				st = int(l[0])
				et = int(l[1])
				dur = int(l[2])
				events['Eblk'].append([st,et,dur])
				blink_flag = 0

			# regular lines will contain tab separated values, beginning with
			# a timestamp, follwed by the values that were asked to be stored
			# in the EDF and a mysterious '...'. Usually, this comes down to
			# timestamp, x, y, pupilsize, ...
			# e.g.: "985288\t  504.6\t  368.2\t 4933.0\t..."
			# NOTE: these values MAY CONTAIN SPACES, but these spaces are
			# ignored by float() (thank you Python!)
			else:
				# see if current line contains relevant data
				try:
					# split by tab
					l = line.split('\t')
					# if first entry is a timestamp, this should work
					int(l[0])
				except:
					message("line '%s' could not be parsed" % line)
					continue # skip this line

				try:
					float(l[1])
				except:
					l[1] = missing
					l[2] = missing
					l[3] = missing
					if len(l) > 5:
						l[4] = missing
						l[5] = missing
						l[6] = missing

				# extract data
				x_l.append(float(l[1]))
				y_l.append(float(l[2]))
				size_l.append(float(l[3]))

				if len(l) > 5:
					try:
						x_r.append(float(l[4]))
						y_r.append(float(l[5]))
						size_r.append(float(l[6]))
					except:
						x_r.append(float(l[1]))
						y_r.append(float(l[2]))
						size_r.append(float(l[3]))

				else:
					x_r.append(float(l[1]))
					y_r.append(float(l[2]))
					size_r.append(float(l[3]))

				time.append(int(l[0])-starttime)
				trackertime.append(int(l[0]))


	# # # # #
	# return

	return data


def read_tobii(filename, start, stop=None, missing=0.0, debug=False):
	"""Returns a list with dicts for every trial.

	Parameters
	----------
	filename : str
		Path to the file that has to be read
	start : str
		Trial start string
	stop : str
		Trial ending string (default = None)
	missing : float
		Value to be used for missing data (default = 0.0)
	debug : bool
		Indicating if DEBUG mode should be on or off; if DEBUG mode is on, information on what the script currently is doing will be printed to the console (default = False)

	Returns
	-------
	data : list
		With a dict for every trial. Following is the dictionary
		0. x -array of Gaze x positions,
		1. y -array of Gaze y positions,
		2. size -array of pupil size,
		3. time -array of timestamps, t=0 at trialstart,
		4. trackertime -array of timestamps, according to the tracker,
		5. events -dict {Sfix, Ssac, Sblk, Efix, Esac, Eblk, msg}

	"""
	return