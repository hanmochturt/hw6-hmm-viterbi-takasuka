import numpy as np
class HiddenMarkovModel:
    """Saves Hidden Markov Model state names and probabilities
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray,
                 prior_probabilities: np.ndarray, transition_probabilities: np.ndarray,
                 emission_probabilities: np.ndarray):
        """Saves Hidden Markov Model state names and probabilities along with useful dictionaries to
        translate between state names and indices

        Args:
            observation_states (np.ndarray): names of each observation state
            hidden_states (np.ndarray): names of each hidden state
            prior_probabilities (np.ndarray): initial probability of observing the hidden states
            transition_probabilities (np.ndarray): probabilities of moving from one hidden state
            to another or repeating the same state; rows are state_1 -> columns are state_2
            emission_probabilities (np.ndarray): probability of moving from a hidden state to an
            observed state; rows are hidden states -> columns are observations
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities