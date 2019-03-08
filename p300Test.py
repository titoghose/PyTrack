import os
import mne
import pickle
import numpy as np 
import matplotlib.pyplot as plt    
import scipy.io as sio  
import scipy.signal as sig
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def smooth(x, window_len):
	"""
	Python implementation of matlab's smooth function
	"""

	if window_len < 3:
		return x

	# Window length must be odd
	if window_len%2 == 0:
		window_len += 1
	
	w = np.ones(window_len)
	y = np.convolve(w, x, mode='valid') / len(w)
	y = np.hstack((x[:window_len//2], y, x[len(x)-window_len//2:]))

	for i in range(0, window_len//2):
		y[i] = np.sum(y[0 : i+i]) / ((2*i) + 1)

	for i in range(len(x)-window_len//2, len(x)):
		y[i] = np.sum(y[i - (len(x) - i - 1) : i + (len(x) - i - 1)]) / ((2*(len(x) - i - 1)) + 1)

	return y

def extract_data(data_file):
    
    if os.path.isfile(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        return data

    sampling_rate = 250
    sampling_interval = sampling_rate / 1000.0

    data = sio.loadmat("/home/upamanyu/Documents/NTU_Creton/Data/kaggle_p300_Datase/P300S01.mat")["data"]

    eeg = np.transpose(data["X"][0][0])
    trial_points = np.squeeze(data["trial"][0][0])
    flash_points = data["flash"][0][0]

    trial_points = np.hstack((trial_points, eeg.shape[1]))

    X = []
    y = []

    cnt = 0
    for trial in trial_points[1:]:
        
        temp_X = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
        temp_y = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0}

        if cnt >= len(flash_points):
            break

        flash = flash_points[cnt][0]
        while flash <= trial:
            
            stimulation = flash_points[cnt][2]
            
            temp_y[stimulation] = flash_points[cnt][3] - 1
            
            if len(temp_X[stimulation]) == 0:
                temp_X[stimulation] = np.array(eeg[:, flash - int(200*sampling_interval) : flash+ int(sampling_interval*600)])
            else:    
                temp_X[stimulation] = (temp_X[stimulation] + np.array(eeg[:, flash - int(200*sampling_interval) : flash + int(sampling_interval*600)])) / 2.
            
            cnt += 1
            if cnt >= len(flash_points):
                break
            flash = flash_points[cnt][0]
        
        X.append(temp_X)
        y.append(temp_y)

    data_X = []
    data_y = []

    for ind, x in enumerate(X):
        for i in x:
            data = []
            for ch in range(8):
                smooth_data = smooth(x[i][ch], 7)
                dec = sig.decimate(smooth_data, q=12)
                data = np.hstack((data, dec))

            data_X.append(data)
            data_y.append(y[ind][i])

    data_X = np.array(data_X)
    data_y = np.array(data_y)

    data = {"X":data_X, "y":data_y}

    with open("kaggle_p300.pickle", "wb") as f:
        pickle.dump(data, f)
    
    return data
    

data_file = 'kaggle_p300.pickle'
data = extract_data(data_file)

kfold = KFold(n_splits=10)

print("LDA Accuracy \t SVM Accuracy \t LSVM Accuracy")

for train_idx, test_idx in kfold.split(data["X"]):
    X_train, y_train = data["X"][train_idx], data["y"][train_idx]
    X_test, y_test = data["X"][test_idx], data["y"][test_idx]
    
    lda_clf = LinearDiscriminantAnalysis()
    svm_clf = SVC(class_weight='balanced', gamma='scale', probability=True)
    lsvm_clf = LinearSVC(class_weight='balanced', dual=False)

    svm_clf.fit(X_train, y_train)
    lsvm_clf.fit(X_train, y_train)
    lda_clf.fit(X_train, y_train)

    print("%2.3f \t\t %2.3f \t\t %2.3f" % (lda_clf.score(X_test, y_test), svm_clf.score(X_test, y_test), lsvm_clf.score(X_test, y_test)))
