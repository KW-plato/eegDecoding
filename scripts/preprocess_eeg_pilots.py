#!/usr/bin/env python3
from ast import literal_eval
from asrpy import ASR
import gc
import json
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import pyxdf
import seaborn as sns

#global section
datasrc = "/net/store/nbp/projects/GTI_decoding/data/pilot"
alltrials_file = "02_general_exp_data/allTrialsData.csv"
experimentdetails_file = "02_general_exp_data/ExperimentDetails.csv"
montagefile = "/net/store/nbp/projects/GTI_decoding/code/pilot/saurabh/eegDecoding/misc/ANT_EEG_channel_mapping.json"

#folder structure: data/pilot/01_subjects_data/P001/eeg/preprocessed/04_PCA
subj_dir = "01_subjects_data"
dir_structure = {
    'events': "eeg/preprocessed/00_events_added",
    'filtered': "eeg/preprocessed/01_filtered",
    'epoched': "eeg/preprocessed/02_epoched",
    'cleaned': "eeg/preprocessed/03_cleaned",
    'pca': "eeg/preprocessed/04_PCA",
    'rawepochs': "eeg/preprocessed/99_uncleanepochs"
    }

trial_types = ["/".join([x,y,z]) for x in ['lift','use'] for y in ['fam','unfam'] for z in ['left','right']]
trial_event_map = {t:i+1 for i,t in enumerate(trial_types)}



#Helper Functions
# change the eeg channel names in xdf to standard channel names using the montage file
def get_channel_names(temp, n):
    ch_names = []
    for i in range(n):
        n = temp[i]['label'][0].replace("ExG", "ch-")
        ch_names.append(montage[n] if "ch-" in n else n)  # only eeg channels are mapped, the aux channels are not
    return ch_names


# save files
def save_file(f, ftype, subj, block):
    save_dir = os.path.join(datasrc, subj_dir, subj, dir_structure[ftype])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if ftype in ['events', 'filtered', 'cleaned']:
        fend = '_eeg.fif'
    else:
        fend = '_epo.fif'

    fname = f"block_{str(block)}{fend}"
    f.save(os.path.join(save_dir, fname), overwrite=True)

    # when saving the epoched mne file also save a numpy version
    if ftype == 'epoched':
        epoch_array = f.get_data()  # convert mne to numpy array
        fname = f"block_{str(block)}_epo.npy"
        with open(os.path.join(save_dir, fname), 'wb') as f:
            np.save(f, epoch_array)


def get_eeg_and_preprocess(df, montage):
    out = 'ok'
    subj = df['subj_id'].item()
    blk = df['block'].item()
    event_streams = literal_eval(df['trig_streams'].item())
    eeg_stream = df['eeg_stream'].item()
    stim_stream = df['stim_stream'].item()
    if (eeg_stream == 99) | (stim_stream == 99) | ('99' in event_streams):
        print(" All datastreams not found. Skipped")
        out = 'skipped'
    else:
        try:
            print("start...")
            data, header = pyxdf.load_xdf(df['filename'].item())  # read the xdf
            eeg_time = data[eeg_stream]["time_stamps"] - data[eeg_stream]["time_stamps"][0]  # get eeg time points
            sr = data[eeg_stream]['info']['effective_srate']  # sampling rate of eeg
            n_ch = int(data[eeg_stream]['info']['channel_count'][0])  # no. of eeg channels
            ch_typ = ['eeg'] * n_ch  # set channel type for mne
            ch_names = get_channel_names(
                data[eeg_stream]['info']['desc'][0]['channels'][0]['channel'],
                n_ch)

            # map the trial start cue i.e. the first index in the event_streams to ecent array for mne
            trials_df = pd.read_csv(os.path.join(datasrc, alltrials_file))
            tdf = trials_df.loc[(trials_df['subj_id'] == subj) &
                                (trials_df['block'] == blk),
                                'condition']
            tdf = tdf.map(trial_event_map)
            x = np.searchsorted(data[eeg_stream]["time_stamps"],
                                data[int(event_streams[0])]["time_stamps"])
            events = np.zeros((x.shape[0], 3), dtype=int)
            events[:, 0] = x
            events[:, 2] = tdf.to_numpy(dtype=int)

            del trials_df, tdf
            gc.collect()

            # create mne info
            info = mne.create_info(ch_names, ch_types=ch_typ, sfreq=sr)

            # mne data array should be in (nChannel,nSamples) whereas xdf stores in (nSamples,nChannel)
            eeg = mne.io.RawArray(np.transpose(data[eeg_stream]["time_series"]), info)
            # drop auxiliaries and not needed channels
            eeg.drop_channels(['BIP65', 'BIP66', 'BIP67', 'BIP68', 'AUX69', 'AUX70', 'AUX71', 'AUX72'])
            # set the montage
            eeg.set_montage('standard_1020')
            # reject the first 20 secs to get rid of voltage swings at the start
            # resample to 256Hz
            # Note: the raw and the event are resampled simultaneously so that they stay more or less in synch.
            eeg.crop(tmin=20)
            eeg_resamp, events_resamp = eeg.resample(sfreq=256, events=events)

            del eeg
            gc.collect()

            # EEG is recorded with average reference. Re-refer to Cz
            eeg_resamp.set_eeg_reference(ref_channels=['Cz'])
            eeg_resamp.drop_channels(['Cz'])
              # save to events folder
            save_file(eeg_resamp, 'events', subj, blk)

            # Filter resampled EEG
            # High pass 0.1 Hz
            # Low pass 120 Hz
            # notch 50Hz,100Hz
            eeg_resamp.filter(l_freq=0.1, h_freq=120)
            eeg_resamp.notch_filter(freqs=[50, 100])
            # save to filtered folder
            save_file(eeg_resamp, 'filtered', subj, blk)

            # epochs paramaters
            tmin = -0.1  # 100 msec before the event boundary
            tmax = 6.0  # each trial is 2.0 + 0.5 + 3.0 + 0.5...NOTE: first 0.5 sec of action is included
            baseline = (tmin, 0)  # i.e. the entire zone from tmin to 0
            epochs = mne.Epochs(eeg_resamp, events_resamp, event_id=trial_event_map, tmin=tmin, tmax=tmax)
            epochs.drop_bad()
            # save to rawepochs folder
            save_file(epochs, 'rawepochs', subj, blk)

            del epochs
            gc.collect()

            # run asr to clean the filtered raw data
            # Apply the ASR
            asr = ASR(sfreq=eeg_resamp.info["sfreq"], cutoff=20)
            asr.fit(eeg_resamp)
            print("ASR fitted\n")
            eeg_resamp = asr.transform(eeg_resamp)
            # save to cleaned folder
            save_file(eeg_resamp, 'cleaned', subj, blk)

            # epoch the cleaned eeg, use the parameter already set-up
            clean_epochs = mne.Epochs(eeg_resamp, events_resamp, event_id=trial_event_map, tmin=tmin, tmax=tmax)
            clean_epochs.drop_bad()
            # save to epoched folder
            save_file(clean_epochs, 'epoched', subj, blk)

            del clean_epochs, eeg_resamp
            gc.collect()

        except Exception as e:
            print(e)
            out = 'failed'

    if out == 'ok':
        print("Success")
    else:
        print(out)

    return out


if __name__ == "__main__":

    with open(montagefile) as f:  # open the montage file provided by Debbie
        montage = json.loads(f.read())
    # change channel names to match standard naming convention in mne standard montage
    montage['ch-1'] = 'Fp1'
    montage['ch-2'] = 'Fpz'
    montage['ch-3'] = 'Fp2'


    expt_df = pd.read_csv(os.path.join(datasrc, experimentdetails_file))
    subjects = expt_df['subj_id'].unique()
    results = []
    for s in subjects:
        subj_df = expt_df[expt_df['subj_id'] == s]
        for block in subj_df['block'].unique():
            print("\nProcessing Subject {} Block {} ##############".format(s, block))
            flag = get_eeg_and_preprocess(subj_df[subj_df['block'] == block], montage)
            results.append(f"Subject: {s} Block: {str(block)} Report: {flag}")
            print("...end")

    print("FINISHED")

    for r in results:
        print(r)



