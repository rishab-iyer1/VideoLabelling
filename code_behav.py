import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_and_clean_files(label_dir, fname):
    """Load and clean subject files, returning valid subject IDs and their data."""
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
    
    return list(txt_files), list(subj_ids)

def parse_transition_file(subj_ids, txt_files, label_dir):
    """Parse a single transition file into a clean DataFrame."""
    all_df = {}
    all_transitions = {}
    skipped = {}
    for idx, (s_id, f) in enumerate(zip(subj_ids, txt_files)):
        f_path = os.path.join(f'{label_dir}', f)
        df = pd.read_csv(f_path, sep='##', engine='python')[2:]  # engine='python' to allow ## separator
        df.columns = ['time', 'info']

        # the transitions of button presses are associated without ## separator, so they show as None
        transitions = df[df['info'].isna()]['time']
        if transitions.empty:
            print(f'No transitions found for {s_id}, excluding')
            skipped[idx] = s_id
            continue
        transitions = transitions.str.split(n=2, expand=True)
        transitions.columns = ['time', 'emotion', 'on_off']

        # convert the time column to just contain time (contained button transitions before)
        df.loc[df['info'].isna(), 'time'] = transitions['time']

        transitions['time'] = transitions['time'].apply(lambda x: round(float(x)))
        all_df[s_id] = df
        all_transitions[s_id] = transitions
    return all_transitions, skipped

def process_emotion_states(transitions_dict, subject_ids, n_trs, emotions_dict):
    """Process emotion states for all subjects."""
    n_subjects = len(subject_ids)
    n_emotions = len(emotions_dict)
    coded_states = np.zeros((n_subjects, n_trs, n_emotions))
    
    for subj_idx, subj_id in enumerate(subject_ids):
        transitions = transitions_dict[subj_id]
        state = np.zeros((n_trs, n_emotions))
        
        # Process each emotion separately
        for emo_idx, (emo_key, emo_name) in enumerate(emotions_dict.items()):
            emo_trans = transitions[(transitions['emotion'] == emo_name) | (transitions['emotion'] == 'All')].sort_values('time')
            
            if not emo_trans.empty:
                for _, trans in emo_trans.iterrows():
                    tr = trans['time']
                    if tr >= n_trs:  # Skip transitions beyond n_trs
                        continue
                    if trans['emotion'] == emo_name:
                        if trans['on_off'] == 'On':
                            state[tr:, emo_idx] = 1
                        elif trans['on_off'] == 'Off':
                            state[tr:, emo_idx] = 0
                    elif trans['emotion'] == 'All' and trans['on_off'] == 'Off':
                        # Handle 'All, Off' transitions
                        next_trans_times = emo_trans[emo_trans['time'] > tr]['time']
                        next_tr = min(next_trans_times.min(), n_trs) if not next_trans_times.empty else n_trs
                        state[tr:next_tr, emo_idx] = 0

        
        # # Process 'All' off transitions last
        # all_trans = transitions[transitions['emotion'] == 'All'].sort_values('time')
        # if not all_trans.empty:
        #     for _, trans in all_trans.iterrows():
        #         if trans['on_off'] == 'Off':
        #             tr = trans['time']
        #             if tr >= n_trs:
        #                 continue
        #             # Find next transition if it exists
        #             next_trans = transitions[transitions['time'] > tr]
        #             if next_trans.empty:
        #                 state[tr:, :] = 0
        #             else:
        #                 next_tr = min(next_trans['time'].min(), n_trs)
        #                 state[tr:next_tr, :] = 0
        
        coded_states[subj_idx] = state
    
    return coded_states

def compute_neutral_states(coded_states):
    """Compute neutral states (when no other emotion is active)."""
    neutral_states = np.all(coded_states == 0, axis=2).astype(int)
    return neutral_states

def smooth_states(states, window_size=6):
    """Apply smoothing to emotion states."""
    kernel = np.ones(window_size) / window_size
    smoothed = np.zeros_like(states)
    for s in range(states.shape[0]):
        for e in range(states.shape[2]):
            smoothed[s, :, e] = np.convolve(states[s, :, e], kernel, 'same')
    return smoothed

def main():
    # Configuration
    TASK = 'onesmallstep'  
    FNAME = 'OneSmallStep' if TASK == 'onesmallstep' else 'ToyStory3'
    N_TRS = 484 if TASK == 'onesmallstep' else 300
    SAVING = True
    PLOTTING = True
    SMOOTH = 'smooth'  # or 'raw'
    
    label_dir = f'/Volumes/BCI/Ambivalent_Affect/VideoRatings/{FNAME}'
    rating_options = {'P': 'Positive', 'N': 'Negative', 'M': 'Mixed', 'Cry': 'Crying'}
    
    # Load and process files
    txt_files, subj_ids = load_and_clean_files(label_dir, FNAME)
    
    # Parse all transition files
    transitions_dict, skipped = parse_transition_file(subj_ids, txt_files, label_dir)
    subj_ids = [s for s in subj_ids if s not in skipped.values()]  # Remove skipped subjects
    for index in sorted(skipped.keys(), reverse=True):  # Remove skipped files (https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time)
        del txt_files[index]  

    assert len(subj_ids) == len(txt_files) == len(transitions_dict)  # important to make sure that txt files and subjects match

    # Process emotion states
    coded_states = process_emotion_states(transitions_dict, subj_ids, N_TRS, rating_options)
    
    # Add neutral states
    neutral_states = compute_neutral_states(coded_states)
    coded_states = np.concatenate([coded_states, neutral_states[:, :, np.newaxis]], axis=2)
    
    # Apply smoothing if requested
    if SMOOTH == 'smooth':
        smoothed_states = smooth_states(coded_states)
        final_states = smoothed_states
    else:
        final_states = coded_states
    
    # Compute statistics
    counts = np.sum(final_states, axis=0)
    proportions = counts / len(subj_ids)
    
    # Plotting
    if PLOTTING:
        plt.figure(figsize=(16, 9))
        colors = ['red', 'blue', 'purple', 'cyan', 'gray']
        emotions = list(rating_options.values()) + ['Neutral']
        
        for i, (emotion, color) in enumerate(zip(emotions, colors)):
            plt.plot(proportions[:, i], color=color, label=emotion)
            
        plt.xlabel('Time')
        plt.ylabel('Proportion of subjects')
        plt.legend()
        plt.title(f'Emotion Consensus: {TASK}')
        
        if SAVING:
            plt.savefig(f'feeling_trend_{TASK}.png')
        plt.show()
    
    # Save results if requested
    if SAVING:
        np.save(f'coded_states_{TASK}.npy', coded_states)
        np.save(f'counts_{TASK}_{SMOOTH}.npy', counts)
        print('Saved counts')

if __name__ == "__main__":
    main()