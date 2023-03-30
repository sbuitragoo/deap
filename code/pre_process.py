from ctypes import Array
from typing import List
import numpy as np
import pandas as pd
from scipy import signal
from db_manager import DEAP_Manager

def get_data_cut(dm: DEAP_Manager, subjects: Array[int], seconds: int):
    fs = 128
    cut_signal = {}
    print(f"Starting data cut of {seconds} seconds for a total of {subjects.shape[0]} subjecs")
    for subject in subjects:
        print(f"Starting cut for subject {subject}")
        subject_data = dm.get_data_for_subject(subject)
        cut_range = seconds * fs
        cut_signal[f"subject{subject}"] = np.zeros((subject_data.shape[0], 32, subject_data.shape[2] - cut_range))
        for trial in range(subject_data.shape[0]):
            cut_signal[f"subject{subject}"][trial, :, :] = subject_data[trial, 32, cut_range:]
    print(f"Data successfully cut!\nTotal of seconds of the new data: {cut_signal['subject1'].shape[2] / fs}\nTotal of channels: {cut_signal.shape[1]}")
    return cut_signal

def filter_data(input_data, subjects: List[int], wl: int, wh: int):
    # Filter between 4 and 45 Hz
    bp_filter = signal.butter(15, [wl,wh], 'bp', fs=128, output='sos')
    
    filtered_signals = {}

    print(f"Starting data filtering...")
    for subject in subjects:
        subject_data = input_data[f"subject{subject}"]
        filtered_signals[f"subject{subject}"] = np.zeros_like(subject_data)
        print(f"Starting filtering for subject {subject}")
        for trial in range(subject_data.shape[0]):
            for ch in range(subject_data.shape[1]):
                filtered_signals[f"subject{subject}"][trial, ch, :] = signal.sosfilt(bp_filter, subject_data[trial, ch, :])

    print(f"Finished data filtering!")
    print(f"Output data shape for subject: {filtered_signals[f'subject{1}'].shape}")
    print(f"Filtered data sample: {filtered_signals[f'subject{1}'][0, :, :]}")
    return filtered_signals

def apply_window(input_data, subjects):
    post_processed_data = {}
    window_size = 128
    overlapping = 0.5

    print(f"Starting windowing with window size of {window_size} and overlapping of {overlapping*100}%")

    for subject in subjects:
        subject_data = input_data[f"subject{subject}"]
        post_processed_data[f"subject{subject}"] = np.zeros((subject_data.shape[0], subject_data.shape[1], int(subject_data.shape[2] / 2)))
        for trial in range(subject_data.shape[0]):
            for i in range(post_processed_data[f"subject{subject}"].shape[2] / window_size):
                if (i == 0):
                    post_processed_data[f"subject{subject}"][trial, :, i:i*window_size] = subject_data[trial, :, i:i*window_size]
                else:
                    post_processed_data[f"subject{subject}"][trial, :, i:i*window_size] = subject_data[trial, :, i*window_size*overlapping:(i*window_size*overlapping)+window_size]
    print(f"Finished windowing")
    return post_processed_data

def split_trial_into_stack(input_data, subjects):
    splitted_data = {}
    window_size = 128

    print(f"Starting trail splitting for each subject, begining with a shape per subject of: {input_data[f'subject{1}'].shape}")

    for subject in subjects:
        subject_data = input_data[f"subject{subject}"]
        splitted_data[f"subject{subject}"] = np.zeros((subject_data.shape[0]*32, subject_data.shape[1], subject_data.shape[2]))
        for original_trial in range(subject_data.shape[0]):
            for new_trial in range(splitted_data[f"subject{subject}"].shape[0]):
                splitted_data[f"subject{subject}"][new_trial, :, :window_size] = subject_data[original_trial, :, new_trial:new_trial*window_size]

    print(f"Finished trail splitting successfully with a new size per subject of: {splitted_data[f'subject{subject}'].shape}")
    return splitted_data
            


def label_preprocessing():
    # Read labels
    labels = pd.read_csv('labels.csv')

    columns = ["Valence", "Arousal", "Dominance", "Liking"]

    # Map labels: label > 5 -> 1 otherwise 0
    for col in columns:
        labels[col] = np.where(labels[col] > 5, 1, 0)

    # Save new labels to a csv file
    labels.to_csv('binary_labels.csv')

def pre_process(db_path: str):
    dm = DEAP_Manager(db_path)
    subjects = np.arange(1,33,1)

    # Load every subject data to subjects dict
    for subject in subjects:
        dm.get_file_for_subject(subject)

    cut_data = get_data_cut(dm, subjects, 3)
    
    print("Sample shape:", cut_data[f"subject{1}"].shape)

    sampled_data = apply_window(cut_data, subjects)

    splitted_data = split_trial_into_stack(sampled_data, subjects)

    print(f"Final shape: {splitted_data[f'subject{1}'].shape}")

    # How the data_preprocessed_python provided by DEAP owners already has been filtered between 4 and 45 Hz, this step is omitted 
    # filtered_data = filter_data(cut_data, subjects, 4, 45)
    
    return splitted_data

if __name__ == "__main__":
    pass
