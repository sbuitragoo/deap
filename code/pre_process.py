import numpy as np
import pickle
import matplotlib.pyplot as plt

class DEAP_PreProcessor: 

    def __init__(self, db):
        self.db = db
        self.subjects = range(1, 33)
        self.file = {}

    def show_sample(self):
        pass

    def show_subject_data_shape(self, subject):
        pass

    def get_file_for_subject(self, subject):
        with open('data_preprocessed_python/s'+ '0' + str(subject) + '.dat', 'rb') as file:
            subject = pickle.load(file, encoding='latin1')
        
        self.file[subject] = file

    def get_data_for_subject(self, subject):
        return self.file[subject]
