"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy as np
from scipy import misc
import math

class HMM(Classifier):
        
    def get_model(self): return None
    def set_model(self, model): pass

    model = property(get_model, set_model)
        
    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function should update self.transtion_count_table
        and self.feature_count_table based on this new given instance.

        Here is a view of the parameter matrices.

        transition_count_table / transition_matrix
                0 (F)   1   2   3
        0 (S)
        1
        2
        3

        emission_count_table / emission_matrix
            0   1   2   3
        0   0   0   0   0
        1
        2
        3

        """

        for instance in instance_list:
            features = instance.features()
            labels = instance.label

            # Start -> First label
            first_label = self.label_index[labels[0]]
            self.transition_count_table[0, first_label] += 1
            # Label label -> Final
            last_label = self.label_index[labels[len(features) - 1]]
            self.transition_count_table[last_label, 0] += 1

            for i in range(len(labels) - 1):
                label = self.label_index[labels[i]]
                next_label = self.label_index[labels[i + 1]]
                self.transition_count_table[label, next_label] += 1
                feature = self.feature_index[features[i]]
                self.feature_count_table[label, feature] += 1
            last_label = self.label_index[labels[len(labels) - 1]]
            last_feature = self.feature_index[features[len(features) - 1]]
            self.feature_count_table[last_label, last_feature] += 1


    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate 
        of the appropriate parameters

        All of the features and transition states are indexed and stored in dictionaries.
        These indices are used for lookup in the parameter matrices.
        """

        features = set([feature for instance in instance_list for feature in instance.features()])
        features.add('UNK')
        labels = set([label for instance in instance_list for label in instance.label])

        self.feature_index = dict((e, i) for (i, e) in enumerate(features))
        self.label_index = dict((e, i + 1) for (i, e) in enumerate(labels))
        self.index_feature = dict((i, e) for (i, e) in enumerate(features))
        self.index_label = dict((i + 1, e) for (i, e) in enumerate(labels))

        self.transition_matrix = np.zeros((len(labels) + 1,len(labels) + 1))
        self.emission_matrix = np.zeros((len(labels) + 1,len(features)))
        self.transition_count_table = np.ones((len(labels) + 1,len(labels) + 1))
        self.transition_matrix[0][0] = 0
        self.feature_count_table = np.ones((len(labels) + 1, len(features)))
        for i in range(self.feature_count_table.shape[1]):
            self.feature_count_table[0][i] = 0

        self._collect_counts(instance_list)

        trans_row_sums = [sum(row) for row in self.transition_count_table]
        feat_row_sums = [sum(row) for row in self.feature_count_table]
        for label in range(self.transition_count_table.shape[0]):
            for next_label in range(self.transition_count_table.shape[1]):
                if (trans_row_sums[label] > 0):
                    self.transition_matrix[label, next_label]\
                        = self.transition_count_table[label, next_label] / trans_row_sums[label]

            for feature in range(self.feature_count_table.shape[1]):
                if (feat_row_sums[label] > 0):
                    self.emission_matrix[label, feature]\
                        = self.feature_count_table[label, feature] / feat_row_sums[label]

    def classify(self, instance):
        """Viterbi decoding algorithm

        Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix

        Returns a list of labels e.g. ['B','I','O','O','B']
        """
        features = instance.features()
        trellis, backtrace_pointers = self.dynamic_programming_on_trellis(instance, False, True)
        # trellis = self.dynamic_programming_on_trellis(instance)
        # beta = self.backward_trellis(instance)

        # Current step in the best sequence
        time = len(features) - 1
        current_step = np.argmax(trellis, axis=0)[time]
        best_sequence = [current_step]
        while time > 0:
            current_step = backtrace_pointers[current_step, time]
            best_sequence = [current_step] + best_sequence
            time -= 1
        best_sequence = [self.index_label[index] for index in best_sequence]
        print(best_sequence)

        return best_sequence


    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        trellis = self.dynamic_programming_on_trellis(instance)
        loglikelihood = 0.0
        return loglikelihood

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True, print_trellis=False):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        Returns trellis filled up with the forward probabilities 
        and backtrace pointers for finding the best sequence

        """
        features = instance.features()
        labels = instance.label

        trellis = np.zeros((len(self.label_index.keys()) + 1,len(features)))
        backtrace_pointers = trellis.copy()
        for label_ind in range(1, trellis.shape[0]):
            if (features[0] in self.feature_index):
                feat_ind = self.feature_index[features[0]]
            else:
                feat_ind = self.feature_index['UNK']
            trellis[label_ind, 0] = self.transition_matrix[0, label_ind]\
                                    * self.emission_matrix[label_ind, feat_ind]

        for time in range(1, len(features)):
            for label_ind in range(1, trellis.shape[0]):
                feat_ind = 0
                if (features[time] in self.feature_index):
                    feat_ind = self.feature_index[features[time]]
                else:
                    feat_ind = self.feature_index['UNK']
                a = self.emission_matrix[label_ind, feat_ind]
                b = [self.transition_matrix[prev, label_ind] * trellis[prev][time - 1]
                    for prev in range(1, trellis.shape[0])]
                if run_forward_alg:
                    trellis[label_ind, time] = a * sum(b)
                else:
                    max_ = max(b)
                    trellis[label_ind, time] = a * max_
                    backtrace_pointers[label_ind, time] = b.index(max_) + 1

        return (trellis, backtrace_pointers)

    def initializeModel(self, instance_list):
        features = set([feature for instance in instance_list for feature in instance.features()])
        features.add('UNK')
        labels = ["B", "I", "O"]
        self.transition_matrix = np.ones((len(labels) + 1,len(labels) + 1))

        for i in range(self.transition_matrix.shape[0]):
            for j in range(self.transition_matrix.shape[1]):
                self.transition_matrix[i, j] /= self.transition_matrix.shape[0]

        self.emission_matrix = np.ones((len(labels) + 1,len(features)))

        for i in range(self.emission_matrix.shape[0]):
            for j in range(self.emission_matrix.shape[1]):
                self.emission_matrix[i, j] /= self.emission_matrix.shape[1]

        self.feature_index = dict((e, i) for (i, e) in enumerate(features))
        self.label_index = dict((e, i + 1) for (i, e) in enumerate(labels))
        self.index_feature = dict((i, e) for (i, e) in enumerate(features))
        self.index_label = dict((i + 1, e) for (i, e) in enumerate(labels))



    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        """Baum-Welch algorithm for fitting HMM from unlabeled data

        The algorithm first initializes the model with the labeled data if given.
        The model is initialized randomly otherwise. Then it runs 
        Baum-Welch algorithm to enhance the model with more data.

        Add your docstring here explaining how you implement this function

        Returns None
        """

        if labeled_instance_list is not None:
            self.train(labeled_instance_list)
        else:
            self.initializeModel(unlabeled_instance_list)

        old_likelihood = 0
        count = 0
        while count < 10:
            likelihood = 0
            for instance in unlabeled_instance_list:
                features = instance.features()
                alpha_table, beta_table = self._run_forward_backward(instance)
                # E-Step
                self.expected_transition_counts = np.ones((self.transition_matrix.shape[0], self.transition_matrix.shape[1]))
                self.expected_feature_counts = np.ones((self.emission_matrix.shape[0], len(features)))

                likelihood += sum([alpha_table[i, alpha_table.shape[1] - 1] * self.transition_matrix[i, 0]
                              for i in range(alpha_table.shape[0])])

                for i in range(self.expected_transition_counts.shape[0]):
                    for j in range(self.expected_transition_counts.shape[1]):
                        for t in range(0, len(features) - 1):
                            feat_ind = self.feature_index[features[t + 1]]
                            l = [alpha_table[i, t], self.transition_matrix[i, j], self.emission_matrix[j, feat_ind], beta_table[j, t + 1]]
                            self.expected_transition_counts[i, j] += misc.logsumexp(l)


                for i in range(self.transition_matrix.shape[0]):
                    denom = sum(self.expected_transition_counts[i, :])
                    for j in range(self.transition_matrix.shape[1]):
                        self.transition_matrix[i, j] = self.expected_transition_counts[i, j] / denom

                featind_gamma = {}
                for j in range(self.expected_feature_counts.shape[0]):
                    featind_gamma[j] = {}
                    for t in range(self.expected_feature_counts.shape[1]):
                        feat_ind = self.feature_index[features[t]]
                        l = [alpha_table[j, t], beta_table[j, t]]
                        self.expected_feature_counts[j, t] = misc.logsumexp(l)

                        if feat_ind not in featind_gamma[j]:
                            featind_gamma[j][feat_ind] = 0
                        featind_gamma[j][feat_ind] += self.expected_feature_counts[j, t]

                for j, it in featind_gamma.items():
                    feat_sum = 0
                    for feat_ind, val in it.items():
                        feat_sum += val
                    for feat_ind, val in it.items():
                        self.emission_matrix[j, feat_ind] = val / feat_sum

            likelihood /= len(unlabeled_instance_list)
            # if self._has_converged(old_likelihood, likelihood):
              #  break
            old_likelihood = likelihood
            count += 1
            print(likelihood)

    def _has_converged(self, old_likelihood, likelihood):
        """Determine whether the parameters have converged or not

        Returns True if the parameters have converged.
        TODO: Find better convergence condition.
        """
        return likelihood == old_likelihood

    def backward_trellis(self, instance):
        features = instance.features()
        labels = instance.label

        trellis = np.zeros((len(self.label_index.keys()) + 1, len(features)))

        for label_ind in range(1, trellis.shape[0]):
            trellis[label_ind, len(features) - 1] = self.transition_matrix[label_ind, 0]

        feat_ind = 0
        for time in range(len(features) - 2, -1, -1):
            for label_ind in range(1, trellis.shape[0]):
                if (features[time + 1] in self.feature_index.keys()):
                    feat_ind = self.feature_index[features[time + 1]]
                else:
                    feat_ind = self.feature_index['UNK']

                trellis[label_ind, time]= sum([self.emission_matrix[prev, feat_ind] * self.transition_matrix[label_ind, prev]
                                               * trellis[prev][time + 1] for prev in range(1, trellis.shape[0])])
        if (features[0] in self.feature_index):
            feat_ind = self.feature_index[features[0]]
        else:
            feat_ind = self.feature_index['UNK']

        # likelihood = sum([trellis[i, 0] * self.transition_matrix[0, i] * self.emission_matrix[i, feat_ind]
        #                  for i in range(1, trellis.shape[0])])

        return trellis


    def _run_forward_backward(self, instance):
        """Forward-backward algorithm for HMM using trellis
    
        Fill up the alpha and beta trellises (the same notation as 
        presented in the lecture and Martin and Jurafsky)
        You can reuse your forward algorithm here

        return a tuple of tables consisting of alpha and beta tables
        """
        alpha_table = self.dynamic_programming_on_trellis(instance)
        beta_table = self.backward_trellis(instance)

        return (alpha_table[0], beta_table)
