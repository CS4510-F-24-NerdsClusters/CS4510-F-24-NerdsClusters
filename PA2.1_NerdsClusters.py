
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
from scipy.stats import skew
from scipy.stats import kurtosis
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

'''
import mne
import os
import pandas as pd
import seaborn as sns
'''
#%% USER INTERFACE              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    sfreq = soi['info']['eeg_info']['effective_srate']

    plt.plot(tStamp,ds_P4)
    plt.title(soi['info']['eeg_info']['channels'][7]['label'])
    plt.xlabel('tStamp')
    plt.ylabel('P4 data')
    plt.grid()
    plt.savefig('OUTPUT//P4.png')
    plt.show()

    plt.plot(tStamp,ds_P3)
    plt.title(soi['info']['eeg_info']['channels'][6]['label'])
    plt.xlabel('tStamp')
    plt.ylabel('P3 data')
    plt.grid()
    plt.savefig('OUTPUT//P3.png')
    plt.show()

    plt.plot(tStamp,ds_Pz)
    plt.title(soi['info']['eeg_info']['channels'][18]['label'])
    plt.xlabel('tStamp')
    plt.ylabel('Pz data')
    plt.grid()
    plt.savefig('OUTPUT//Pz.png')
    plt.show()

    def filter(data, num, sampling_freq): #bandstop filter (notch and impedance)
        low = (num - 1)/(sampling_freq/2)
        high = (num + 1)/(sampling_freq/2)
        b,a = scipy.signal.butter(N = 2, Wn = [low,high], btype = 'bandstop')
        return scipy.signal.filtfilt(b,a,data)
    #
    def bandpass_filter(data, low_freq, high_freq, sampling_freq):
        low = low_freq/(sampling_freq/2)
        high = high_freq/(sampling_freq/2)
        b, a = scipy.signal.butter(N = 4, Wn = [low,high], btype = 'bandpass')
        return scipy.signal.filtfilt(b, a, data)
    #
    def run_filters(data):
        filtered_data = filter(data, 60, sfreq)
        filtered_data = filter(filtered_data, 120, sfreq)
        filtered_data = filter(filtered_data, 180, sfreq)
        filtered_data = filter(filtered_data, 240, sfreq)
        filtered_data = filter(filtered_data, 125, sfreq)
        filtered_data = bandpass_filter(filtered_data, .5, 32, sfreq)
        return filtered_data
    #
    def rereference(data):
        signal_avg = np.mean(data, axis = None)
        rereferenced_data = data - signal_avg
        return rereferenced_data

    filtered_data_P4 = run_filters(ds_P4)
    filtered_data_P3 = run_filters(ds_P3)
    filtered_data_Pz = run_filters(ds_Pz)

    plt.plot(tStamp,ds_P4)
    plt.plot(tStamp,filtered_data_P4)
    plt.title('P4 original and filtered')
    plt.xlabel('tStamp')
    plt.ylabel('data')
    plt.grid()
    plt.savefig('OUTPUT//P4withfilters.png')
    plt.show()

    plt.plot(tStamp,ds_P3)
    plt.plot(tStamp,filtered_data_P3)
    plt.title('P3 original and filtered')
    plt.xlabel('tStamp')
    plt.ylabel('data')
    plt.grid()
    plt.savefig('OUTPUT//P3withfilters.png')
    plt.show()

    plt.plot(tStamp,ds_Pz)
    plt.plot(tStamp,filtered_data_Pz)
    plt.title('Pz original and filtered')
    plt.xlabel('tStamp')
    plt.ylabel('data')
    plt.grid()
    plt.savefig('OUTPUT//Pzwithfilters.png')
    plt.show()

    #Apply re-referencing
    P4_data_rereferenced = rereference(filtered_data_P4)
    P3_data_rereferenced = rereference(filtered_data_P3)
    Pz_data_rereferenced = rereference(filtered_data_Pz)
 
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(filtered_data_P4, label='P4 (original)', alpha=0.7)
    plt.plot(filtered_data_P3, label='P3 (original)', alpha=0.7)
    plt.plot(filtered_data_Pz, label='Pz (original)', alpha=0.7)
    plt.title('Original EEG Signals')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(P4_data_rereferenced, label='P4 (re-referenced)', alpha=0.7)
    plt.plot(P3_data_rereferenced, label='P3 (re-referenced)', alpha=0.7)
    plt.plot(Pz_data_rereferenced, label='Pz (re-referenced)', alpha=0.7)
    plt.title('Re-referenced EEG Signals')
    plt.legend()

    plt.tight_layout()
    plt.show()

    subjects = ['sb1','sb2']
    sessions = ['se1','se2']
    pathSoIRoot = 'INPUT\\DataSmall\\'
    dataSe1 = []
    dataSe2 = []
    window_size = 100
    for subject in subjects:
        for session in sessions:
            pathSoi = f'{pathSoIRoot}{subject}\\{session}\\'
            with os.scandir(pathSoi) as files:
                for file in files:
                    means = []
                    stds = []
                    with open(file,'rb') as fp:
                        sbfile = pckl.load(fp)
                    p4data = run_filters(sbfile['series'][7])
                    p3data = run_filters(sbfile['series'][6])
                    pzdata = run_filters(sbfile['series'][18])

                    for start in range(0, len(p4data) - window_size + 1):
                        data_section = p4data[start:start + window_size]
                        mean = np.mean(data_section)
                        std = np.std(data_section)
                        means.append(mean)
                        stds.append(std)

                    for start in range(0, len(p3data) - window_size + 1):
                        data_section = p3data[start:start + window_size]
                        mean = np.mean(data_section)
                        std = np.std(data_section)
                        means.append(mean)
                        stds.append(std)

                    for start in range(0, len(pzdata) - window_size + 1):
                        data_section = pzdata[start:start + window_size]
                        mean = np.mean(data_section)
                        std = np.std(data_section)
                        means.append(mean)
                        stds.append(std)
                
                    f1 = np.mean(means)
                    f2 = np.std(means)
                    f3 = skew(means)
                    f4 = kurtosis(means)
                    f5 = np.mean(stds)
                    f6 = np.std(stds)
                    f7 = skew(stds)
                    f8 = kurtosis(stds)
                    if session == 'se1':
                        dataSe1.append([subject, session, f1, f2, f3, f4, f5, f6, f7, f8])
                    else:
                        dataSe2.append([subject, session, f1, f2, f3, f4, f5, f6, f7, f8])

    df1 = pd.DataFrame(dataSe1, columns = ['subject', 'session', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'])
    df2 = pd.DataFrame(dataSe2, columns = ['subject', 'session', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'])

    df1.to_csv('OUTPUT\\TrainValidateData.csv')
    df2.to_csv('OUTPUT\\TestData.csv')

    #features only
    df1_features = df1.iloc[:, 2:]

    pca = PCA(n_components = 2)
    df1PCA = pd.DataFrame(pca.fit_transform(df1_features), columns=['PC1','PC2'])

    #KMeans
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(df1PCA)
    df1['KMeansCluster'] = kmeans.labels_

    plt.scatter(df1PCA['PC1'], df1PCA['PC2'], c=df1['KMeansCluster'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering Results')
    plt.savefig('OUTPUT//KmeansCluster.png')
    plt.show()

    #hierarchical
    linkage_matrix = linkage(df1_features, method = 'ward')
    cluster_labels = fcluster(linkage_matrix, t=10, criterion = 'distance')
    df1['HierarchicalCluster'] = cluster_labels
    plt.scatter(df1PCA['PC1'], df1PCA['PC2'], c=df1['HierarchicalCluster'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Cluster')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Hierarchical Clustering Results')
    plt.savefig('OUTPUT//HierarchicalCluster.png')
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
