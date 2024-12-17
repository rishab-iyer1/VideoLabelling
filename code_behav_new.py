"""
Take txt file for each subject and code the behavioral responses into an Excel sheet
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

TASK = 'onesmallstep'  # 'toystory'
if TASK == 'onesmallstep':
    N_TRS = 484
    FNAME = 'OneSmallStep'
elif TASK == 'toystory':
    N_TRS = 300
    FNAME = 'ToyStory3'
else:
    raise ValueError('invalid task')

SAVING = False
PLOTTING = True
SMOOTH = 'smooth'
WINDOW_SIZE = 30
STEP_SIZE = 5
N_WINDOWS = int((N_TRS - WINDOW_SIZE) / STEP_SIZE) + 1

label_dir = f'/Volumes/BCI/Ambivalent_Affect/VideoRatings/{FNAME}'
emotions = ['P', 'N', 'M', 'X', 'Cry', 'P_smooth', 'N_smooth', 'M_smooth', 'X_smooth', 'Cry_smooth']

# get all txt files
txt_files = os.listdir(f'{label_dir}')
txt_files = [x for x in txt_files if x.endswith('.txt')]
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

all_df = {}
all_transitions = {}
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

# now that we have all the data - arrange them into array and code the emotions to the nearest TR
subj_ids = [s for s in subj_ids if s not in skipped]
coded_df = np.empty(shape=(len(subj_ids), N_TRS, len(emotions)))
coded_df[:] = np.nan
# get the times that fall between marking on and off for each emotion
rating_options = {'P': 'Positive', 'N': 'Negative', 'M': 'Mixed', 'Cry': 'Crying', 'All': 'All'}
emo_to_label = {'P': 'Positive', 'N': 'Negative', 'M': 'Mixed', 'X': 'Neutral', 'Cry': 'Crying'}
for s, s_id in enumerate(subj_ids):
    for e, emo in enumerate(rating_options.keys()):
        BUTTON = 0
        for tr in range(N_TRS):
            trans = all_transitions[s_id][all_transitions[s_id]['emotion'] == rating_options[emo]]
            if not trans.empty: assert trans.emotion.nunique() == 1  # assert that all emotion transitions are the same, if they exist on this iteration
            for i in trans.index:
                t = trans[trans.index == i]['time'].item()
                if tr == t:
                    if trans[trans['time'] == tr].on_off.iloc[-1] == 'On':  # iloc instead of item() for indexing in edge case where there is more than one transition for the current TR; we can just pick the last one
                        BUTTON = 1
                    elif trans[trans['time'] == tr].on_off.iloc[-1] == 'Off':
                        BUTTON = 0
                    else:
                        raise ValueError('Transition not "On" or "Off"')

                    # if not trans.empty and trans['emotion'].iloc[0] == 'All':  # override the previous button switch if
                    #     # on the final pass, we want to update the labels based on the 'All off' marker; there are 2 cases
                    #     # 1. there is some emotion transition after "All off"
                    #     #     a. in this case, we want to zero out the values between the TR where "All off" was marked and the next transition
                    #     #     b. zero out values for all emotions until we run into an "on" marker for any emotion
                    #     # 2. "All off" is the last transition
                    #     #     a. zero out all subsequent values
                    #
                    #     all_off_start_tr = trans[trans['time'] == tr]['time'].item()
                    #     idx = trans.index[trans['time'] == all_off_start_tr].item()  # index of all_off_start_tr
                    #
                    #     if idx == all_transitions[s_id]['time'].index[-1].item():  # this "All off" is the last transition
                    #         all_off_end_tr = False
                    #     else:  # There is a transition after this "All off"
                    #         end_idx = np.where(all_transitions[s_id]['time'].index == idx)[0].item() + 1  # The index of the transition after this "All off"
                    #         all_off_end_tr = all_transitions[s_id]['time'].iloc[end_idx]  # The TR at which to stop zeroing
                    #
                    #     if all_off_end_tr:
                    #         if all_off_start_tr <= tr < all_off_end_tr:
                    #             # we are in the case where there is some emotion transition after "All off", so zeroing ends at all_off_end_tr
                    #             BUTTON = 0
                    #     else:
                    #         if tr >= all_off_start_tr:
                    #             # we are in the case where "All off" is the last transition; everything from here on should be zero
                    #             BUTTON = 0

                coded_df[s, tr, e] = BUTTON

# neutral is coded as 1 if no other emotion is coded in the same TR
# (all zeros in columns P, N, M, Cry)
coded_df[:, :, 3] = np.all(coded_df[:, :, [0, 1, 2, 4]] == 0, axis=2).astype(int)

# smooth button presses from columns (P, N, M, X, Cry)
# into columns (P_smooth, N_smooth, M_smooth, X_smooth, Cry_smooth)
for s in range(len(subj_ids)):
    for e in range(5):
        coded_df[s, :, e + 5] = np.convolve(coded_df[s, :, e], np.ones(6), 'same') / 6

if SMOOTH == 'raw':
    coded_df = coded_df[:, :, :5]
elif SMOOTH == 'smooth':
    coded_df = coded_df[:, :, 5:]
else:
    raise ValueError('smooth not specified')

counts = np.empty(shape=coded_df.shape[1:])
for e in range(coded_df.shape[-1]):
    counts[:, e] = np.sum(coded_df[:, :, e], axis=0)

if SAVING:
    np.save(f'counts_{TASK}_{SMOOTH}.npy', counts)
    print('saved counts')

test = counts / len(subj_ids)
plt.figure(figsize=(16, 9))
colors = ['red', 'blue', 'purple', 'gray', 'cyan']
for i in range(test.shape[1]):
    plt.plot(test[:, i], color=colors[i])
plt.xlabel('Time')
plt.ylabel('Proportion of subjects')
plt.legend(emo_to_label.values())
plt.title(f'Emotion Consensus: {TASK}')
if SAVING:
    plt.savefig(f'feeling_trend')
if PLOTTING:
    plt.show()

slide_behav = np.empty(shape=(N_WINDOWS, coded_df.shape[-1]))
for i in range(N_WINDOWS):
    slide_behav[i] = np.mean(counts[i * STEP_SIZE:i * STEP_SIZE + N_WINDOWS], axis=0)

if SAVING:
    np.save(f'{label_dir}/slide_behav_{TASK}_{SMOOTH}.npy', slide_behav)
    print('saved')
