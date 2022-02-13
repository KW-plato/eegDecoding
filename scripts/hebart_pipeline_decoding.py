#Decoding Task and Object from eeg
#Reference: Hebart et al. eLife 2018;7:e32816. DOI: https://doi.org/10.7554/eLife.32816

import gc
import glob
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import re
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#declarations
#folder structure: data/pilot/01_subjects_data/P001/eeg/preprocessed/04_PCA
#datasrc = r"C:\Users\saubi\TechSpace\data"
datasrc = "/net/store/nbp/projects/GTI_decoding/data/pilot"
dir_struct = {
    'dir_subj': "01_subjects_data",
    'dir_gen': "02_general_exp_data",
    'filtered': "eeg/preprocessed/01_filtered",
    'epoched': "eeg/preprocessed/02_epoched",
    'cleaned': "eeg/preprocessed/03_cleaned",
    'pca': "analysis",
    'rawepochs': "eeg/preprocessed/99_epochs_raw"
    }

subjects = ['P001','P002','P004','S002','S005']
num_turns = 500
tmin= 0.1
fwhm = 15
sigma = fwhm / np.sqrt(8 * np.log(2))
factors = {'task':{1:'use',2:'lift'}, 'object':{1:'fam',2:'unfam'},'orientation':{1:'left',2:'right'}}
baselines ={'task':(0,.1),'object':(2.5,2.6),'action':(5.5,5.6)}
trial_types = ["/".join([x,y,z]) for x in ['lift','use'] for y in ['fam','unfam'] for z in ['left','right']]
trial_event_map = {t:i+1 for i,t in enumerate(trial_types)}
colors = ["blue","green", "red", "cyan", "magenta", "yellow", "orange", "steelblue"]
color_map = {i+1:c for i,c in enumerate(colors)}

if __name__ == "__main__":
    subject_accuracies = []
    for s in subjects:
        pca_loc = os.path.join(datasrc, dir_struct['dir_subj'], dir_struct['pca'], f"{s}_epo.fif")
        pca = mne.read_epochs(pca_loc, preload=False, verbose=False)
        pca_data = pca.get_data()
        for factor in ['task', 'object']:
            conditions = factors[factor]
            tmin, tmax = baselines[factor]
            baseline_start = round(pca.info['sfreq'] * tmin)
            baseline_end = round(pca.info['sfreq'] * tmax)
            mean = np.mean(pca_data[:, :, baseline_start:baseline_end], axis=2)
            std = np.std(pca_data[:, :, baseline_start:baseline_end], axis=2)
            mean_subtracted = pca_data - mean[:, :, None]
            std_adjusted = mean_subtracted / std[:, :, None]
            output = gaussian_filter1d(std_adjusted, sigma, axis=2)
            pca_epoch = mne.EpochsArray(output,
                                        info=mne.create_info(output.shape[1], pca.info['sfreq'], ch_types='eeg'),
                                        tmin=-0.1,
                                        events=pca.events,
                                        event_id=pca.event_id,
                                        verbose=False)
            conds = factors[factor]
            accuracies = np.zeros((pca_data.shape[2], num_turns, 1))
            for x in range(num_turns):
                cond_matrix = {1: [], 2: []}
                train_x = {}
                train_y = {}
                test_x = {}
                test_y = {}
                for cond in conds.keys():
                    sel_epchs = pca_epoch[conds[cond]].get_data()
                    idx_arr = np.arange(sel_epchs.shape[0])
                    np.random.shuffle(idx_arr)
                    i = 0
                    store = []
                    while i + 10 < len(idx_arr):
                        random_idx = idx_arr[i:i + 10]
                        fam_random = sel_epchs[random_idx]
                        fam_random = np.average(fam_random, axis=0)
                        fam_random = np.transpose(fam_random)
                        cond_matrix[cond].append(fam_random[:, None, :])
                        i = i + 10
                    train_x[cond] = np.concatenate(cond_matrix[cond][:-1], axis=1)
                    train_y[cond] = [cond] * train_x[cond].shape[1]
                    test_x[cond] = cond_matrix[cond][-1]
                    test_y[cond] = cond
                y_train = []
                y_test = []
                for cond in [1, 2]:
                    y_train.extend(train_y[cond])
                    y_test.append(test_y[cond])
                for t in range(pca_data.shape[2]):
                    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                    X_train = np.concatenate([train_x[cond][t] for cond in [1, 2]])
                    X_test = np.concatenate([test_x[cond][t] for cond in [1, 2]])
                    clf.fit(X_train, y_train)
                    acc = accuracy_score(y_test, clf.predict(X_test))
                    accuracies[t, x] = acc
                if x % 100 == 0:
                    print("Subject: {}, factor: {} epoch: {}".format(s,factor,x))

            final_acc = np.mean(accuracies, axis=1)
            # plt.plot(final_acc)
            # plt.show()
            tmpdf = pd.DataFrame(final_acc)
            tmpdf = tmpdf.reset_index().rename(columns={'index': 't', 0: 'accuracy'})
            tmpdf['subj_idx'] = s
            tmpdf['factor'] = factor
            subject_accuracies.append(tmpdf)


    alldata = pd.concat(subject_accuracies)
    alldata.reset_index(drop=True, inplace=True)
    alldata.to_csv(os.path.join(datasrc, dir_struct['dir_subj'], dir_struct['pca'], "analysis.csv"),index=False,header=True)
    alldata['t']=alldata['t']/256 - .1
    eventlines ={'task':0,'object':2.5,'action':5.5}
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.lineplot(data=alldata, x="t", y="accuracy", hue='factor', ax=ax)
    ax.axhline(0.5,color='red')
    for i in baselines.keys():
        ax.axvline(eventlines[i],color='gray')
        ax.text(eventlines[i],0.2,i,fontsize=14,rotation=90)
    plt.savefig(os.path.join(datasrc, dir_struct['dir_subj'], dir_struct['pca'], "analysis.jpg"))


