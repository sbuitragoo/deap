import numpy as np
import pickle
import matplotlib.pyplot as plt

class DEAP_Manager: 

    def __init__(self, db_path):
        self.db_path = db_path
        self.subjects = range(1, 33)
        self.file = {}

    def show_sample(self):
        pass

    def show_subject_data_shape(self, subject):
        pass

    def get_file_for_subject(self, subject):
        with open(f'{self.db_path}/s'+ (('0' + str(subject)) if subject < 10 else str(subject)) + '.dat', 'rb') as file:
            subject = pickle.load(file, encoding='latin1')
        
        self.file[subject] = subject

    def get_data_for_subject(self, subject):
        return self.file[subject]['data']
    
    def get_labels_for_subject(self, subject):
        return self.file[subject]['labels']
    
    def get_subject_shape(self, subject):
        return self.file[subject].shape
