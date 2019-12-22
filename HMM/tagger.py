import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here
	TL = len(tags)
	TRL = len(train_data)
	# state_dict
	state_dict = {}

	for i in range(TL):
		state_dict[tags[i]] = i

	# obs_dict
	obs_dict = {}
	pi_util = np.zeros(len(state_dict))
	A_util = np.zeros([len(state_dict),len(state_dict)])

	for i in range(TRL):
		pi_util[ state_dict[train_data[i].tags[0]] ] += 1
		for j in range(len(train_data[i].words)):
			if j != len(train_data[i].words)-1:
				A_util[state_dict[train_data[i].tags[j]],state_dict[train_data[i].tags[j+1]]] += 1
			if train_data[i].words[j] not in obs_dict:
				obs_dict[ train_data[i].words[j] ] = len(obs_dict)

	B_util = np.zeros([len(state_dict),len(obs_dict)])
	for i in range(TRL):
		for j in range(len(train_data[i].words)):
			B_util[state_dict[train_data[i].tags[j]], obs_dict[train_data[i].words[j]]] += 1

	pi = pi_util / sum(pi_util)
	A = A_util / np.sum(A_util, axis=1).reshape((len(state_dict),1))
	B = B_util / np.sum(B_util, axis=1).reshape((len(state_dict),1))


	model = HMM(pi, A, B, obs_dict, state_dict)
	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	for i in range(len(test_data)):
		for j in range(len(test_data[i].words)):
			if test_data[i].words[j] not in model.obs_dict:
				model.obs_dict[test_data[i].words[j]] = len(model.obs_dict)
				small = np.ones([len(model.B), 1])*pow(10,-6)
				model.B = np.concatenate((model.B, small), axis=1)
		tagging.append(model.viterbi(test_data[i].words))
	###################################################
	return tagging
