import unittest
import numpy as np
import pandas as pd
from io import StringIO
from code_behav_v2 import process_emotion_states, compute_neutral_states, parse_transition_file

class TestEmotionCoding(unittest.TestCase):
    def create_test_transitions(self, transition_data):
        """Helper to create transitions DataFrame from string data"""
        df = pd.read_csv(StringIO(transition_data), sep=',')
        df['time'] = df['time'].astype(float).round().astype(int)
        return df

    def test_single_emotion_basic(self):
        """Test basic on/off transitions for a single emotion"""
        transitions = self.create_test_transitions("""
time,emotion,on_off
0,Positive,On
5,Positive,Off
""")
        
        transitions_dict = {'test_subject': transitions}
        result = process_emotion_states(transitions_dict, ['test_subject'], 10, {'P': 'Positive'})
        
        expected = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1s from 0-4, 0s from 5-9
        ]).reshape(1, 10, 1)
        
        np.testing.assert_array_equal(result, expected)

    def test_multiple_emotions_no_overlap(self):
        """Test multiple emotions with non-overlapping periods"""
        transitions = self.create_test_transitions("""
time,emotion,on_off
0,Positive,On
3,Positive,Off
4,Negative,On
7,Negative,Off
""")
        
        transitions_dict = {'test_subject': transitions}
        result = process_emotion_states(
            transitions_dict, 
            ['test_subject'], 
            10, 
            {'P': 'Positive', 'N': 'Negative'}
        )
        
        expected = np.zeros((1, 10, 2))
        expected[0, 0:3, 0] = 1  # Positive
        expected[0, 4:7, 1] = 1  # Negative
        
        np.testing.assert_array_equal(result, expected)

    def test_multiple_emotions_with_overlap(self):
        """Test multiple emotions with overlapping periods"""
        transitions = self.create_test_transitions("""
time,emotion,on_off
0,Positive,On
5,Negative,On
7,Positive,Off
9,Negative,Off
""")
        
        transitions_dict = {'test_subject': transitions}
        result = process_emotion_states(
            transitions_dict, 
            ['test_subject'], 
            12, 
            {'P': 'Positive', 'N': 'Negative'}
        )
        
        expected = np.zeros((1, 12, 2))
        expected[0, 0:7, 0] = 1  # Positive
        expected[0, 5:9, 1] = 1  # Negative
        
        np.testing.assert_array_equal(result, expected)

    def test_all_off_transition(self):
        """Test the 'All' off functionality"""
        transitions = self.create_test_transitions("""
time,emotion,on_off
0,Positive,On
2,Negative,On
5,All,Off
7,Positive,On
""")
        
        transitions_dict = {'test_subject': transitions}
        result = process_emotion_states(
            transitions_dict, 
            ['test_subject'], 
            10, 
            {'P': 'Positive', 'N': 'Negative'}
        )
        
        expected = np.zeros((1, 10, 2))
        expected[0, 0:5, 0] = 1  # Positive until All off
        expected[0, 2:5, 1] = 1  # Negative until All off
        expected[0, 7:, 0] = 1   # Positive after new On
        
        np.testing.assert_array_equal(result, expected)

    def test_neutral_state_computation(self):
        """Test computation of neutral states"""
        # Create a known state array
        states = np.zeros((1, 10, 4))  # One subject, 10 timepoints, 4 emotions
        states[0, 0:3, 0] = 1  # Positive for first 3 timepoints
        states[0, 5:7, 1] = 1  # Negative for timepoints 5-6
        
        neutral = compute_neutral_states(states)
        
        expected = np.zeros((1, 10))
        expected[0, 3:5] = 1  # Neutral when no emotions are active
        expected[0, 7:] = 1
        
        np.testing.assert_array_equal(neutral, expected)

    def test_multiple_transitions_same_tr(self):
        """Test handling of multiple transitions at the same timepoint"""
        transitions = self.create_test_transitions("""
time,emotion,on_off
5,Positive,On
5,Positive,Off
5,Positive,On
""")
        
        transitions_dict = {'test_subject': transitions}
        result = process_emotion_states(
            transitions_dict, 
            ['test_subject'], 
            10, 
            {'P': 'Positive'}
        )
        
        expected = np.zeros((1, 10, 1))
        expected[0, 5:, 0] = 1  # Should use last transition (On)
        
        np.testing.assert_array_equal(result, expected)

    def test_out_of_bounds_transitions(self):
        """Test handling of transitions beyond N_TRS"""
        transitions = self.create_test_transitions("""
time,emotion,on_off
0,Positive,On
12,Positive,Off
""")
        
        transitions_dict = {'test_subject': transitions}
        result = process_emotion_states(
            transitions_dict, 
            ['test_subject'], 
            10, 
            {'P': 'Positive'}
        )
        
        expected = np.ones((1, 10, 1))  # Should stay on until end of timepoints
        
        np.testing.assert_array_equal(result, expected)

def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmotionCoding)
    unittest.TextTestRunner(verbosity=3).run(suite)

if __name__ == '__main__':
    run_tests()