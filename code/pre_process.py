import numpy as np
import pandas as pd
from scipy import signal
from db_manager import DEAP_Manager

def data_preprocessing():
    dm = DEAP_Manager()
    subjects = range(33)

    # Load every subject data to subjects dict
    for subject in subjects:
        dm.get_file_for_subject(subject)
    
    # Filter between 4 and 45 Hz
    bp_filter = signal.butter(15, [4,45], 'bp', fs=128, output='sos')
    
    filtered_signals = {}

    for subject in subjects:
        subject_data = dm.get_data_for_subject(subject)
        filtered_signals[f"subject{subject}"] = np.zeros_like(subject_data)
        for trial in range(subject_data.shape[0]):
            filtered_signals[f"subject{subject}"][trial, :, :] = signal.sosfilt(bp_filter, subject_data[trial, :, :])


if __name__ == "__main__":
    # Read labels
    labels = pd.read_csv('labels.csv')

    columns = ["Valence", "Arousal", "Dominance", "Liking"]

    # Map labels: label > 5 -> 1 otherwise 0
    for col in columns:
        labels[col] = np.where(labels[col] > 5, 1, 0)

    # Save new labels to a csv file
    labels.to_csv('binary_labels.csv')

