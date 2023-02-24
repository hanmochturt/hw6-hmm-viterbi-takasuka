import copy
import numpy as np
class ViterbiAlgorithm:
    """Determines the most probable hidden state sequence based on set probabilities and an input
    observed state sequence
    """    

    def __init__(self, hmm_object):
        """Collect prior, transition, and emission probabilities, hidden and observation state
        names, and dictionaries that index state names

        Args:
            hmm_object (_type_): probabilities, state names, and index state name dictionaries
        """              
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """Find the most probable hidden state sequence based on known prior, transition,
        and emission probabilities

        Args:
            decode_observation_states (np.ndarray): observation state sequence

        Returns:
            np.ndarray: most probable sequence of hidden states
        """        
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]
        best_path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))
        max_hidden_state_probabilities = np.zeros((len(decode_observation_states),
                         len(self.hmm_object.hidden_states)))

        # Compute initial delta:
        # 1. Calculate the product of the prior and emission probabilities. This will be used to decode the first observation state.
        # 2. Scale      
        delta = np.multiply(np.reshape(self.hmm_object.prior_probabilities, (-1, 1)),
                            self.hmm_object.emission_probabilities)  # hi1, hi1 -> oi
        first_obs_index = self.hmm_object.observation_states_dict[decode_observation_states[0]]
        max_hidden_state_probabilities[0, :] = delta[:, first_obs_index]
        delta = np.reshape(delta[:, first_obs_index], (-1, 1))  # from each hidden state
        #delta = delta / np.sum(delta)

        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):
            # TODO: comment the initialization, recursion, and termination steps

            # hi1 -> o1, hi1 -> hi2
            product_of_delta_and_transition_emission = np.multiply(delta,
                                                                   self.hmm_object.transition_probabilities)
            # hi1 -> o1, hi1 -> hi2, hi2 -> oi2
            this_obs_index = self.hmm_object.observation_states_dict[decode_observation_states[trellis_node]]
            emission_this_observation = np.reshape(self.hmm_object.emission_probabilities[:,
                                        this_obs_index], (1, -1))
            #print(emission_this_observation, "emission this obs")
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition_emission,
                                                                   emission_this_observation)


            # Update delta and scale
            # hi1 -> o1, hi1 -> hi2, hi2 -> o2
            delta = np.reshape(np.amax(product_of_delta_and_transition_emission, axis=0), (-1, 1))
            #delta = delta / np.sum(delta)
            # Select the hidden state sequence with the maximum probability
            max_probabilities_hidden = np.amax(product_of_delta_and_transition_emission, axis=0)
            max_probabilities_indices = np.argmax(product_of_delta_and_transition_emission, axis=0)
            path[trellis_node, :] = max_probabilities_indices
            max_hidden_state_probabilities[trellis_node, :] = max_probabilities_hidden

            # Update best path
            for hidden_state in range(len(self.hmm_object.hidden_states)):
                break
            # Set best hidden state sequence in the best_path np.ndarray THEN copy the best_path to path
            #path = best_path.copy()
        # Select the last hidden state, given the best path (i.e., maximum probability)
        print(max_hidden_state_probabilities)
        last_max_hidden_probability_index = int(np.argmax(max_hidden_state_probabilities[-1, :]))
        best_path_hidden_indices_list = [last_max_hidden_probability_index]
        previous_hidden_index = last_max_hidden_probability_index
        print(path)
        for t in range(len(decode_observation_states)-2, -1, -1):
            best_path_hidden_indices_list.insert(0, path[t+1, previous_hidden_index])
            previous_hidden_index = int(path[t+1, previous_hidden_index])


        best_hidden_state_path = np.array(best_path_hidden_indices_list)
        best_hidden_state_path = np.vectorize(self.hmm_object.hidden_states_dict.get)(best_hidden_state_path)
        print(best_hidden_state_path)
        return best_hidden_state_path