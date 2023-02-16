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

    def get_file_for_subject(self, sub):
        with open(f'{self.db_path}/s'+ (('0' + str(sub)) if sub < 10 else str(sub)) + '.dat', 'rb') as file:
            subject = pickle.load(file, encoding='latin1')
            
        self.file[f"subject{sub}"] = subject

    def get_data_for_subject(self, subject):
        return self.file[f"subject{subject}"]['data']
    
    def get_labels_for_subject(self, subject):
        return self.file[f"subject{subject}"]['labels']
    
    def get_subject_shape(self, subject):
        return self.file[f"subject{subject}"].shape
