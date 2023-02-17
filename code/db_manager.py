import numpy as np
import pickle
import matplotlib.pyplot as plt

class DEAP_Manager: 

    def __init__(self, db_path):
        self.db_path = db_path
        self.subjects = range(1, 33)
        self.file = {}
        self.processed_file = {}

    def show_sample(self, sub):
        subject = self.file[f"subject{sub}"]['data']
        time = np.linspace(0, subject.shape[-1] / 128, 8064) # time samples / sample freq
        plt.figure(figsize=(16,9), dpi=90)
        plt.xlabel("time[s]")
        
        for ch in range(subject.shape[1]):
            plt.plot(subject[0, ch, :], time)
            plt.grid()
        

    def get_file_for_subject(self, sub):
        with open(f'{self.db_path}/s'+ (('0' + str(sub)) if sub < 10 else str(sub)) + '.dat', 'rb') as file:
            subject = pickle.load(file, encoding='latin1')
            
        self.file[f"subject{sub}"] = subject

    def get_data_for_subject(self, sub):
        return self.file[f"subject{sub}"]['data']
    
    def get_labels_for_subject(self, sub):
        return self.file[f"subject{sub}"]['labels']
    
    def get_subject_shape(self, sub):
        return (self.file[f"subject{sub}"]['data'].shape, self.file[f"subject{sub}"]['labels'].shape)
    
    def get_wanted_channels(self, channels):
        if (len(channels) == 0):
            raise Exception("Channels list should have at least one element")    
        
        existant_keys = list(self.file.keys())
        input_shape = self.file[existant_keys[0]]['data'].shape

        new_data = np.zeros((input_shape[0], len(channels), input_shape[-1]))

        for index in range(len(channels)):
            new_data[:,index,:] = channels[:,index,:]

        return new_data
