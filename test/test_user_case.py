"""
UCSF BMI203: Biocomputing Algorithms
Author: Hannah Takasuka
Date: Feb 24, 2023
Program: Oral Sciences
Description: Testing hidden markov models on made up data about student commitment, being on
time, early pregnancy, and my mood when I come home
"""
import pytest
import numpy as np
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from src.models.hmm import HiddenMarkovModel
from src.models.decoders import ViterbiAlgorithm


def test_use_case_lecture():
    """It is hypothesized that the variance in whether a grad student is committed to ambivalent
    can be explained by the rates of R01 vs R21 funding
    """
    # index annotation observation_states=[i,j]
    observation_states = ['committed',
                          'ambivalent']  # A graduate student's dedication to their rotation lab

    # index annotation hidden_states=[i,j]
    hidden_states = ['R01',
                     'R21']  # The NIH funding source of the graduate student's rotation project

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_one_data['prior_probabilities'],
                                         # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_one_data['transition_probabilities'],
                                         # transition_probabilities[:,hidden_states[i]]
                                         use_case_one_data[
                                             'emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities,
                       use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities,
                       use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities,
                       use_case_one_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm
    assert use_case_one_hmm.prior_probabilities.shape == (len(observation_states),)
    assert use_case_one_hmm.transition_probabilities.shape == (
    len(hidden_states), len(hidden_states))
    assert use_case_one_hmm.emission_probabilities.shape == (
    len(hidden_states), len(observation_states))

    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


def test_user_case_one():
    """It is hypothesized that the variance in whether someone is late or on-time can be
    explained by the rates of traffic
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                                         use_case_one_data['prior_probabilities'],
                                         # prior probabilities of hidden states in the order specified in the hidden_states list
                                         use_case_one_data['transition_probabilities'],
                                         # transition_probabilities[:,hidden_states[i]]
                                         use_case_one_data[
                                             'emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

    # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities,
                       use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities,
                       use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities,
                       use_case_one_hmm.emission_probabilities)

    # Check HMM dimensions and ViterbiAlgorithm
    assert use_case_one_hmm.prior_probabilities.shape == (len(observation_states),)
    assert use_case_one_hmm.transition_probabilities.shape == (len(hidden_states), len(hidden_states))
    assert use_case_one_hmm.emission_probabilities.shape == (len(hidden_states), len(observation_states))

    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states ==
                      np.array(['no-traffic', 'no-traffic', 'traffic',
                                'traffic', 'traffic', 'no-traffic']))


def test_user_case_two():
    """It is hypothesized that the variance in whether a birth is early or healthy can be
    explained by the rates of smoking in mothers
    """
    observation_states = ['early birth', 'healthy birth']
    hidden_states = ['smoker', 'nonsmoker']
    observation_sequence = [observation_states[0], observation_states[0], observation_states[0],
                            observation_states[1]]

    prior = np.array([0.072, 1-0.072])
    transition = np.array([[0.7, 1-0.7], [0.1, 0.9]])
    emission = np.array([[0.6, 1-0.6], [0.05, 1-0.05]])

    use_case_hmm = HiddenMarkovModel(observation_states,
                                     hidden_states,
                                     prior,
                                     transition,
                                     emission)
    use_case_one_viterbi = ViterbiAlgorithm(use_case_hmm)
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        observation_sequence)
    assert np.alltrue(use_case_decoded_hidden_states == np.array([hidden_states[0], hidden_states[0],
                                                   hidden_states[0], hidden_states[1]]))


def test_user_case_three():
    """It is hypothesized that the variance in my mood when I arrive home can be
    explained by whether my PI gave me free food during lab meeting
    """
    observation_states = ['sad', 'neutral', 'happy']
    hidden_states = ['no food', 'food']
    observation_sequence = [observation_states[2], observation_states[1], observation_states[0],
                            observation_states[1]]

    prior = np.array([0.3, 0.7])
    transition = np.array([[0.7, 1-0.7], [0.1, 0.9]])
    emission = np.array([[0.6, 0.3, 0.1], [0.1, 0.3, 0.6]])

    use_case_hmm = HiddenMarkovModel(observation_states,
                                     hidden_states,
                                     prior,
                                     transition,
                                     emission)
    use_case_one_viterbi = ViterbiAlgorithm(use_case_hmm)
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(
        observation_sequence)
    assert np.alltrue(use_case_decoded_hidden_states == np.array([hidden_states[1],
                                                                  hidden_states[1],
                                                                  hidden_states[1],
                                                                  hidden_states[1]]))
