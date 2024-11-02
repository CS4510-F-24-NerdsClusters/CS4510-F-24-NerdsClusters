
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

'''
import mne
import numpy  as np 
import os
import pandas as pd
import seaborn as sns
'''
#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
subject = 'sb1'
session = 'se1'
pathSoIRoot = 'INPUT\\stream\\'
pathSoi = f'{pathSoIRoot}{subject}\\{session}\\'
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
    print(soi['info']['eeg_info']['channels'][7]['label'])
    print(soi['info']['eeg_info']['channels'][6]['label'])
    print(soi['info']['eeg_info']['channels'][18]['label'])
    ds_P4 = soi['series'][7]
    ds_P3 = soi['series'][6]
    ds_Pz = soi['series'][18]
    tStamp = soi['tStamp']
    print(soi['info']['eeg_info']['effective_srate'])

    plt.plot(tStamp,ds_P4)
    plt.title(soi['info']['eeg_info']['channels'][7]['label'])
    plt.xlabel = 'tStamp'
    plt.ylabel = 'P4 data'
    plt.grid()
    plt.savefig('OUTPUT//P4.png')
    plt.show()

    plt.plot(tStamp,ds_P3)
    plt.title(soi['info']['eeg_info']['channels'][6]['label'])
    plt.xlabel = 'tStamp'
    plt.ylabel = 'P3 data'
    plt.grid()
    plt.savefig('OUTPUT//P3.png')
    plt.show()

    plt.plot(tStamp,ds_Pz)
    plt.title(soi['info']['eeg_info']['channels'][18]['label'])
    plt.xlabel = 'tStamp'
    plt.ylabel = 'Pz data'
    plt.grid()
    plt.savefig('OUTPUT//Pz.png')
    plt.show()
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
