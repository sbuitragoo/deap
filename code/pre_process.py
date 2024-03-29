from ctypes import Array
from typing import List
import numpy as np
import pandas as pd
from scipy import signal
import argparse
from db_manager import DEAP_Manager

def get_data_cut(dm: DEAP_Manager, subjects: Array[int], seconds: int):
    fs = 128
    cut_signal = {}
    print("#---------------------------------------------------------------------------------------------------------------------#")
    print(f"Starting data cut of {seconds} seconds for a total of {subjects.shape[0]} subjecs")
    for subject in subjects:
        print(f"Starting cut for subject {subject}")
        subject_data = dm.get_data_for_subject(subject)
        cut_range = seconds * fs
        cut_signal[f"subject{subject}"] = np.zeros((subject_data.shape[0], 32, subject_data.shape[2] - cut_range))
        for trial in range(subject_data.shape[0]):
            cut_signal[f"subject{subject}"][trial, :, :] = subject_data[trial, :32, cut_range:]
    print(f"Data successfully cut!\nTotal of seconds of the new data: {cut_signal['subject1'].shape[2] / fs}\nTotal of channels: {cut_signal['subject1'].shape[1]}")
    return cut_signal

def filter_data(input_data, subjects: List[int], wl: int, wh: int):
    # Filter between 4 and 45 Hz
    bp_filter = signal.butter(15, [wl,wh], 'bp', fs=128, output='sos')
    
    filtered_signals = {}

    print("#---------------------------------------------------------------------------------------------------------------------#")
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

    print("#---------------------------------------------------------------------------------------------------------------------#")
    print(f"Starting windowing with window size of {window_size} and overlapping of {overlapping*100}%")

    for subject in subjects:
        subject_data = input_data[f"subject{subject}"]
        post_processed_data[f"subject{subject}"] = np.zeros((subject_data.shape[0], subject_data.shape[1], int(subject_data.shape[2] / 2) + window_size*2))
        for trial in range(subject_data.shape[0]):
            for i in range(int(post_processed_data[f"subject{subject}"].shape[2] / window_size)):
                # print(f"Making windowing of trial {trial} and second: {i}")
                if (i == 0):
                    post_processed_data[f"subject{subject}"][trial, :, i:(i+1)*window_size] = subject_data[trial, :, i:(i+1)*window_size]
                else:
                    post_processed_data[f"subject{subject}"][trial, :, i*window_size:(i*window_size)+window_size] = subject_data[trial, :, int(i*window_size*overlapping):int(i*window_size*overlapping)+window_size]
    print(f"Finished windowing")
    return post_processed_data

def split_trial_into_stack(input_data, subjects):
    splitted_data = {}
    window_size = 128

    print("#---------------------------------------------------------------------------------------------------------------------#")
    print(f"Starting trail splitting for each subject, begining with a shape per subject of: {input_data[f'subject{1}'].shape}")

    for subject in subjects:
        subject_data = input_data[f"subject{subject}"]
        splitted_data[f"subject{subject}"] = np.zeros((int(subject_data.shape[0]*(subject_data.shape[2] / window_size)), subject_data.shape[1], window_size))
        print(f"Starting spliting for Subject {subject}")
        flattened_trials = np.zeros((1,subject_data.shape[1],int(subject_data.shape[0]*subject_data.shape[2])))
        for original_trail in range(subject_data.shape[0]):
            if original_trail == 0:
                flattened_trials[0, :,original_trail:(original_trail+1)*subject_data.shape[2]] = subject_data[original_trail, :, :]
            else:
                flattened_trials[0, :,original_trail*subject_data.shape[2]:(original_trail*subject_data.shape[2])+subject_data.shape[2]] = subject_data[original_trail, :, :]

        for new_trail in range(splitted_data[f"subject{subject}"].shape[0]):
            if (new_trail == 0):
                # print(f"Out shape: {new_trail,(new_trail+1)*window_size}")
                splitted_data[f"subject{subject}"][new_trail, :, :window_size] = flattened_trials[0, :, new_trail:(new_trail+1)*window_size]
            else:
                # print(f"Out shape: {new_trail*window_size,(new_trail*window_size)+window_size}")
                splitted_data[f"subject{subject}"][new_trail, :, :window_size] = flattened_trials[0, :, new_trail*window_size:(new_trail*window_size)+window_size]
                    
    print(f"Finished trail splitting successfully with a new size per subject of: {splitted_data[f'subject{subject}'].shape}")
    print("#---------------------------------------------------------------------------------------------------------------------#")
    return splitted_data

def normalize(input_data, subjects):
    print("Starting normalizing...")
    normalized_data = {}
    max_value_per_trial_per_channel = []
    for subject in subjects:
        subject_data = input_data[f"subject{subject}"]
        for trail in range(subject_data.shape[0]):
            for ch in range(subject_data.shape[1]):
                max_value_per_trial_per_channel.append(np.max(subject_data[trail, ch, :]))

    max_value = np.max(max_value_per_trial_per_channel)

    print("Max value: ", max_value)
    
    for subject in subjects:
        print(f"Normalizing data por subject {subject}")
        subject_data = input_data[f"subject{subject}"]
        normalized_data[f"subject{subject}"] = np.zeros_like(subject_data)
        for trail in range(subject_data.shape[0]):
            for ch in range(subject_data.shape[1]):
                normalized_data[f"subject{subject}"][trail, ch, :] = subject_data[trail, ch, :]/max_value

    print(f"Finished normalizing")
    print("#---------------------------------------------------------------------------------------------------------------------#")

    return normalized_data


def label_binarization():
    # Read labels
    labels = pd.read_csv('labels.csv')

    columns = ["Valence", "Arousal", "Dominance", "Liking"]

    # Map labels: label > 5 -> 1 otherwise 0
    for col in columns:
        labels[col] = np.where(labels[col] > 5, 1, 0)

    # Save new labels to a csv file
    labels.to_csv('binary_labels.csv', index=False)

def label_preprocessing():
    column_names = ["Participant_id", "Trial", "Valence", "Arousal", "Dominance", "Liking"]
    binary_labels = pd.read_csv("binary_labels.csv")
    new_labels = np.zeros((40960, 6)) # 1280 trails x 32 subjects, 4 labels

    print(f"Starting label mapping...")

    for index, label in enumerate(column_names):
        for previous_trail in range(40*32): # 40 trails x 32 subjects 
            if previous_trail == 0:
                new_labels[previous_trail:(previous_trail+1)*32, index] = binary_labels[label].iloc[previous_trail].astype(int)
            else:
                new_labels[previous_trail*32:(previous_trail+1)*32, index] = binary_labels[label].iloc[previous_trail].astype(int)
    
    print(f"Label mapping finished!")

    new_labels_df = pd.DataFrame(new_labels, columns=column_names, dtype="Int64")

    print(f"Starting file writing...")
    new_labels_df.to_csv('final_labels.csv', index=False)
    print(f"File successfully written!")

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

    normalized_data = normalize(splitted_data, subjects)

    print(f"Final shape: {normalized_data[f'subject{1}'].shape}")

    # How the data_preprocessed_python provided by DEAP owners already has been filtered between 4 and 45 Hz, this step is omitted 
    # filtered_data = filter_data(cut_data, subjects, 4, 45)
    
    return normalized_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    params = subparser.add_parser('params')
    params.add_argument('--db', type=str, required=True,
                        help="Path to the data base")
    
    arguments = parser.parse_args()

    if arguments.command == "params":
        pre_processed_data = pre_process(arguments.db)
    else:
        pre_processed_data = pre_process("../../DEAP")
