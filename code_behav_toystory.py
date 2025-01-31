# Take txt file for each subject and code the behavioral responses into an Excel sheet

import os
import pandas as pd
import numpy as np
import xarray as xr

task = 'Toy_Story'
n_TRs = 300
emotions = ['P', 'N', 'M', 'X', 'Cry', 'P_smooth', 'N_smooth', 'M_smooth', 'X_smooth', 'Cry_smooth']
# Set working directory
label_dir = '/Volumes/BCI/Ambivalent_Affect/fMRI_Study/VideoLabelling/Toy_Story_Labelled'

# get all txt files
txt_files = os.listdir(f'{label_dir}')
try:
    subj_ids = [x.split('-')[1] for x in txt_files]
except IndexError:
    print('Some files are not in the correct format')
    raise

# rename pilot# to P##
for i, s in enumerate(subj_ids):
    if 'pilot' in s:
        print(f'Renaming {subj_ids[i]} to {subj_ids[i].replace("pilot", "P")}')
        subj_ids[i] = subj_ids[i].replace('pilot', 'P')
        s = subj_ids[i]
    # acceptable formats are P#, N#, VR#, VR##
    if not any([s.startswith(x) for x in ['P', 'N', 'VR']]):
        print(f'Invalid ID: {subj_ids[i]}, removing from text and subject list')
        subj_ids.pop(i)
        txt_files.pop(i)

txt_files.sort(key=lambda x: x.lower())
subj_ids.sort(key=lambda x: x.lower())

all_df = dict()
all_transitions = dict()
skipped = []
for s_id, f in zip(subj_ids, txt_files):
    f_path = os.path.join(f'{label_dir}', f)
    df = pd.read_csv(f_path, sep='##', engine='python')[2:]  # engine='python' to allow ## separator
    df.columns = ['time', 'info']

    # the transitions of button presses are associated without ## separator, so they show as None
    transitions = df[df['info'].isna()]['time']
    if transitions.empty:
        print(f'No transitions found for {s_id}, excluding')
        skipped.append(s_id)
        continue
    transitions = transitions.str.split(n=2, expand=True)
    transitions.columns = ['time', 'emotion', 'on_off']

    # convert the time column to just contain time (contained button transitions before)
    df.loc[df['info'].isna(), 'time'] = transitions['time']

    transitions['time'] = transitions['time'].apply(lambda x: round(float(x)))
    all_df[s_id] = df
    all_transitions[s_id] = transitions

# now that we have all the data - arrange them into the xarray and code the emotions to the nearest TR
subj_ids = [s for s in subj_ids if s not in skipped]
coded_df = np.empty(shape=(len(subj_ids), n_TRs, len(emotions)))
coded_df[:] = np.nan
# get the times that fall between marking on and off for each emotion
emo_to_label = {'P': 'Positive', 'N': 'Negative', 'M': 'Mixed', 'X': 'Neutral', 'Cry': 'Crying'}
for s, s_id in enumerate(subj_ids):
    for e, emo in enumerate(emotions[:5]):
        button = 0
        for tr in range(n_TRs):
            trans = all_transitions[s_id][all_transitions[s_id]['emotion'] == emo_to_label[emo]]
            for t in trans['time']:
                if tr == t:
                    if button == 0:
                        button = 1
                    elif button == 1:
                        button = 0
            coded_df[s, tr, e] = button

# neutral is coded as 1 if no other emotion is coded in the same TR (all zeros in columns P, N, M, Cry)
coded_df[:, :, 3] = np.all(coded_df[:, :, [0, 1, 2, 4]] == 0, axis=2).astype(int)

# smooth button presses from columns (P, N, M, X, Cry) into columns (P_smooth, N_smooth, M_smooth, X_smooth, Cry_smooth)
for s in range(len(subj_ids)):
    for e in range(5):
        coded_df[s, :, e + 5] = np.convolve(coded_df[s, :, e], np.ones(6), 'same') / 6


coded_df = xr.DataArray(coded_df, dims=['subj_id', 'TR', 'emotion'],
                        coords={'subj_id': subj_ids, 'TR': np.arange(n_TRs), 'emotion': emotions})
# save coded_df
coded_df.to_netcdf(f'{label_dir}/../coded_df_{task}.nc')
