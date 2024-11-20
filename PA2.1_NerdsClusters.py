
#%% MODULE BEGINS
module_name = '<***>'

'''
Version: <***>

Description:
    <***>

Authors:
    Chloe Catanese
    Rowan Merritt

Date Created     :  <***>
Date Last Updated:  <***>

Doc:
    <***>

Notes:
    <***>
'''

#%% IMPORTS                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
   import os
   #os.chdir("./../..")
#

#custom imports


#other imports
from copy import deepcopy as dpcpy
import pickle as pckl
from matplotlib import pyplot as plt
import scipy
import numpy as np

'''
import mne
import os
import pandas as pd
import seaborn as sns
'''
#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
subjects = ['sb1','sb2']
sessions = ['se1','se2']
pathSoIRoot = 'INPUT\\DataSmall\\'
for subject in subjects:
    for session in sessions:
        pathSoi = f'{pathSoIRoot}{subject}\\{session}\\'
        with os.scandir(pathSoi) as files:
            for file in files:
                print(file.name)

soi_file = '1_132_bk_pic.pckl'

#Load SoI object
with open("INPUT\\1_132_bk_pic.pckl", 'rb') as fp:
    soi = pckl.load(fp)
#%% CONSTANTS                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% CONFIGURATION               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#%% INITIALIZATIONS             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#%% DECLARATIONS                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Global declarations Start Here



#Class definitions Start Here



#Function definitions Start Here
def main():
    return
    # print(soi['info']['eeg_info']['channels'][7]['label'])
    # print(soi['info']['eeg_info']['channels'][6]['label'])
    # print(soi['info']['eeg_info']['channels'][18]['label'])
    # ds_P4 = soi['series'][7]
    # ds_P3 = soi['series'][6]
    # ds_Pz = soi['series'][18]
    # tStamp = soi['tStamp']
    # sfreq = soi['info']['eeg_info']['effective_srate']

    # plt.plot(tStamp,ds_P4)
    # plt.title(soi['info']['eeg_info']['channels'][7]['label'])
    # plt.xlabel('tStamp')
    # plt.ylabel('P4 data')
    # plt.grid()
    # plt.savefig('OUTPUT//P4.png')
    # plt.show()

    # plt.plot(tStamp,ds_P3)
    # plt.title(soi['info']['eeg_info']['channels'][6]['label'])
    # plt.xlabel('tStamp')
    # plt.ylabel('P3 data')
    # plt.grid()
    # plt.savefig('OUTPUT//P3.png')
    # plt.show()

    # plt.plot(tStamp,ds_Pz)
    # plt.title(soi['info']['eeg_info']['channels'][18]['label'])
    # plt.xlabel('tStamp')
    # plt.ylabel('Pz data')
    # plt.grid()
    # plt.savefig('OUTPUT//Pz.png')
    # plt.show()

    # def filter(data, num, sampling_freq): #bandstop filter (notch and impedance)
    #     low = (num - 1)/(sampling_freq/2)
    #     high = (num + 1)/(sampling_freq/2)
    #     b,a = scipy.signal.butter(N = 2, Wn = [low,high], btype = 'bandstop')
    #     return scipy.signal.filtfilt(b,a,data)
    # #
    # def bandpass_filter(data, low_freq, high_freq, sampling_freq):
    #     low = low_freq/(sampling_freq/2)
    #     high = high_freq/(sampling_freq/2)
    #     b, a = scipy.signal.butter(N = 4, Wn = [low,high], btype = 'bandpass')
    #     return scipy.signal.filtfilt(b, a, data)
    # #

    # filtered_data_P4 = filter(ds_P4, 60, sfreq)
    # filtered_data_P4 = filter(filtered_data_P4, 120, sfreq)
    # filtered_data_P4 = filter(filtered_data_P4, 180, sfreq)
    # filtered_data_P4 = filter(filtered_data_P4, 240, sfreq)
    # filtered_data_P4 = filter(filtered_data_P4, 125, sfreq)
    # filtered_data_P4 = bandpass_filter(filtered_data_P4, .5, 32, sfreq)

    # filtered_data_P3 = filter(ds_P3, 60, sfreq)
    # filtered_data_P3 = filter(filtered_data_P3, 120, sfreq)
    # filtered_data_P3 = filter(filtered_data_P3, 180, sfreq)
    # filtered_data_P3 = filter(filtered_data_P3, 240, sfreq)
    # filtered_data_P3 = filter(filtered_data_P3, 125, sfreq)
    # filtered_data_P3 = bandpass_filter(filtered_data_P3, .5, 32, sfreq)

    # filtered_data_Pz = filter(ds_Pz, 60, sfreq)
    # filtered_data_Pz = filter(filtered_data_Pz, 120, sfreq)
    # filtered_data_Pz = filter(filtered_data_Pz, 180, sfreq)
    # filtered_data_Pz = filter(filtered_data_Pz, 240, sfreq)
    # filtered_data_Pz = filter(filtered_data_Pz, 125, sfreq)
    # filtered_data_Pz = bandpass_filter(filtered_data_Pz, .5, 32, sfreq)

    # plt.plot(tStamp,ds_P4)
    # plt.plot(tStamp,filtered_data_P4)
    # plt.title('P4 original and filtered')
    # plt.xlabel('tStamp')
    # plt.ylabel('data')
    # plt.grid()
    # plt.savefig('OUTPUT//P4withfilters.png')
    # plt.show()

    # plt.plot(tStamp,ds_P3)
    # plt.plot(tStamp,filtered_data_P3)
    # plt.title('P3 original and filtered')
    # plt.xlabel('tStamp')
    # plt.ylabel('data')
    # plt.grid()
    # plt.savefig('OUTPUT//P3withfilters.png')
    # plt.show()

    # plt.plot(tStamp,ds_Pz)
    # plt.plot(tStamp,filtered_data_Pz)
    # plt.title('Pz original and filtered')
    # plt.xlabel('tStamp')
    # plt.ylabel('data')
    # plt.grid()
    # plt.savefig('OUTPUT//Pzwithfilters.png')
    # plt.show()

    # #Apply re-referencing
    # original_P4_signal = filtered_data_P4
    # original_P3_signal = filtered_data_P3
    # original_Pz_signal = filtered_data_Pz

    # P4_signal_avg = np.mean(original_P4_signal, axis=None)
    # P3_signal_avg = np.mean(original_P3_signal, axis=None)
    # Pz_signal_avg = np.mean(original_Pz_signal, axis=None)

    # P4_data_referenced = original_P4_signal - P4_signal_avg
    # P3_data_referenced = original_P3_signal - P3_signal_avg
    # Pz_data_referenced = original_Pz_signal - Pz_signal_avg
 
    # plt.figure(figsize=(12, 6))

    # plt.subplot(2, 1, 1)
    # plt.plot(original_P4_signal, label='P4 (original)', alpha=0.7)
    # plt.plot(original_P3_signal, label='P3 (original)', alpha=0.7)
    # plt.plot(original_Pz_signal, label='Pz (original)', alpha=0.7)
    # plt.title('Original EEG Signals')
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(P4_data_referenced, label='P4 (re-referenced)', alpha=0.7)
    # plt.plot(P3_data_referenced, label='P3 (re-referenced)', alpha=0.7)
    # plt.plot(Pz_data_referenced, label='Pz (re-referenced)', alpha=0.7)
    # plt.title('Re-referenced EEG Signals')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
#

#%% MAIN CODE                  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main code start here


#%% SELF-RUN                   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Main Self-run block
if __name__ == "__main__":
    
    print(f"\"{module_name}\" module begins.")
    #TEST Code
    main()
# %%
