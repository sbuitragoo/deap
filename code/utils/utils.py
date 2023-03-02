import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from db_manager import DEAP_Manager

def show_db_samples():
    db_path = '../../data_preprocessed_python'

    deap_manager = DEAP_Manager(db_path)

    subjects = range(32)
    
    random_subject = np.random.randint(0, len(subjects))

    #Load data for a random subject
    deap_manager.get_file_for_subject(random_subject)


    #Show a random sample of the whole channels for the first trial of a random subject
    deap_manager.show_sample(subjects[random_subject])

    #Show every channel across time for the same random subject between the 40 trials
    random_subject_data = deap_manager.get_data_for_subject(random_subject)

    plt.figure(figsize=(20,14), dpi=90)
    time = np.linspace(0, random_subject_data.shape[-1]/128, random_subject_data.shape[-1])
    plt.xlabel("Time [s]")
    plt.ylabel("Channels")

    for channel in range(random_subject_data.shape[1]):
        plt.subplot(random_subject_data.shape[1], 1, channel+1)
        for trial in range(random_subject_data.shape[0]):
            plt.plot(time, random_subject_data[trial, channel, :])
            plt.grid(True)
    
    print(f"Data size: {deap_manager.get_subject_shape(random_subject)}")

if __name__ == "__main__":
    pass