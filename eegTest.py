import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import ICA
from autoreject import AutoReject, get_rejection_threshold, Ransac

# sfreq = 256
# times = np.arange(0, 10, 0.001)

# columns = ["StimulusName", "EventSource", "Af3 (Epoc)", "F7 (Epoc)", "F3 (Epoc)", "FC5 (Epoc)", "T7 (Epoc)", "P7 (Epoc)", "O1/Pz (Epoc)", "O2 (Epoc)", "P8 (Epoc)", "T8 (Epoc)", "FC6 (Epoc)", "F4 (Epoc)", "F8 (Epoc)", "AF4 (Epoc)"]

# df = pd.read_csv('/home/upamanyu/Documents/NTU/csv_files_with_column_headers/Guilty Subject 11.csv', usecols=columns, low_memory=False)
# df2 = pd.read_csv('/home/upamanyu/Documents/NTU/csv_files_with_column_headers/Innocent Subject 11.csv', usecols=columns, low_memory=False)

# eeg_indices = np.where((df.EventSource.str.contains("Raw EEG")))[0]
# df = df.iloc[eeg_indices, :]
# indices = np.where(df.StimulusName.str.contains("Wallet"))[0]

# stim_val = pd.factorize(df.StimulusName)[0]
# stim_names = pd.factorize(df.StimulusName)[1]

# stim_014 = np.zeros(len(eeg_indices))
# stim_014[indices] = 1#stim_val[indices]

# data = np.array(df.iloc[:, 2:].copy())
# data = np.transpose(data)
# data = np.vstack((data, stim_014))

# eeg_indices = np.where((df2.EventSource.str.contains("Raw EEG")))[0]
# df2 = df2.iloc[eeg_indices, :]
# indices = np.where(df2.StimulusName.str.contains("Wallet"))[0]

# stim_val = pd.factorize(df2.StimulusName)[0]
# stim_names = pd.factorize(df2.StimulusName)[1]

# stim_014 = np.zeros(len(eeg_indices))
# stim_014[indices] = 1#stim_val[indices]

# data2 = np.array(df2.iloc[:, 2:].copy())
# data2 = np.transpose(data2)
# data2 = np.vstack((data2, stim_014))

# ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'stim']
# ch_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4", "STIM 014"]

# info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
# raw = mne.io.RawArray(data, info)
# raw2 = mne.io.RawArray(data2, info)
# picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

montage = mne.channels.read_montage('standard_1020')
for i in montage:
	print(i)

# # raw.plot(events=events, scalings='auto', show=False, n_channels=14, title="Before")
# # raw.plot_psd(area_mode='range', show=False, picks=picks, average=False)
# # raw.notch_filter(np.arange(60, 121, 60), picks=picks, filter_length='auto', phase='zero')
# raw.filter(None, 30., fir_design='firwin')
# raw.filter(1., None, fir_design='firwin')
# raw2.filter(None, 30., fir_design='firwin')
# raw2.filter(1., None, fir_design='firwin')
# # raw.plot_psd(show=False)
# # raw.plot(block=True, scalings='auto', show=False, n_channels=14, title="After")
# # plt.show()
# events = mne.find_events(raw, initial_event=True)
# events2 = mne.find_events(raw2, initial_event=True)
# event_id = {"Wallet":1}

# epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=1.0, proj=True, picks=picks, preload=False)
# epochs2 = mne.Epochs(raw2, events2, event_id, tmin=0, tmax=1.0, proj=True, picks=picks, preload=False)
# # print(epochs)

# # reject = get_rejection_threshold(epochs)
# # print(reject)

# # ar = AutoReject(cv=7, picks=picks)
# # epochs = ar.transform(epochs)

# evoked = epochs['Wallet'].average()
# plt.plot(epochs["Wallet"].get_data())

# evoked2 = epochs2['Wallet'].average()
# plt.plot(epochs2["Wallet"].get_data())

# plt.show()


# # n_components = 14
# # method = 'fastica'

# # ica = ICA(n_components=n_components, method=method, max_iter=1000)
# # ica.fit(epochs, reject=reject)
# # ica.plot_overlay(inst=raw)

# # # # ica.plot_overlay(inst=raw)
# # # ica.plot_components(picks=range(14))