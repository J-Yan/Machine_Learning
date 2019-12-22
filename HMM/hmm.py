from __future__ import print_function
import numpy as np


class HMM:


    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def get_key(self, val):
        for key, value in self.state_dict.items():
            if val == value:
                return key

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        # alpha[i, 1]
        for i in range(S):
            alpha[i, 0] = self.pi[i] * self.B[i, self.obs_dict[Osequence[0]]]

        # alpha[i, t]
        for t in range(1, L):
            for i in range(S):
                alpha[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * sum(self.A[:,i] * alpha[:,t-1])
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        # beta[i, L] == 1
        for i in range(S):
            beta[i, L-1] = 1

        # other beta
        for t in range(2, L+1):
            for i in range(S):
                beta[i, L-t] = sum(self.A[i,:] * self.B[:,self.obs_dict[Osequence[L-t+1]]] * beta[:, L-t+1])

        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = sum(alpha[:, len(Osequence)-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        P = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        for i in range(S):
            for t in range(L):
                prob[i, t] = alpha[i, t] * beta[i, t] / P

        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        P = self.sequence_prob(Osequence)
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        for i in range(S):
            for j in range(S):
                for t in range(L - 1):
                    prob[i, j ,t] = (alpha[i, t] * self.A[i, j] * self.B[j, self.obs_dict[Osequence[t+1]]] * beta[j, t+1]) / P
        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        sig = np.zeros([S, L])
        delta = np.zeros([S, L])

        for i in range(S):
            sig[i, 0] = self.pi[i] * self.B[i, self.obs_dict[Osequence[0]]]

        for t in range(1,L):
            for i in range(S):
                sig[i, t] = self.B[i, self.obs_dict[Osequence[t]]] * max(self.A[:, i] * sig[:, t-1])
                delta[i, t] = np.argmax(self.A[:, i] * sig[:, t-1])

        path.insert(0, np.argmax(sig[:,L-1]))

        for i in range(1,L):
            path.insert(0, delta[int(path[0]), L-i])
        for i in range(L):
            path[i] = self.get_key(path[i])


        ###################################################
        return path
