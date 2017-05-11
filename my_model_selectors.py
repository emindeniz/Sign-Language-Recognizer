import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def select(self):
        """ select the best model for self.this_word based on
               BIC score for n between self.min_n_components and self.max_n_components

               :return: GaussianHMM object
               """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        model_and_scores = []

        for current_n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            try:
                hmm_model = GaussianHMM(n_components=current_n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)
                #here we calculate the number of free parameters in the model
                variance = np.array([np.diag(hmm_model.covars_[i]) for i in range(hmm_model.n_components)])
                n_params = (len(hmm_model.transmat_)*(len(hmm_model.transmat_)-1)+
                                        current_n_components*len(hmm_model.means_[0])+
                                        current_n_components*len(variance[0]))

                BIC_score = -2*logL + n_params*np.log(len(self.X))
                model_and_scores.append([hmm_model,BIC_score] )

                if self.verbose:
                    print("model created for {} with {} states and score {} and logL {}".format(self.this_word, current_n_components,BIC_score,logL))

            except:
                #since model failed it must have infinite BIC score and it will never be selected
                model_and_scores.append([None, float("inf")])
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, current_n_components))



        best_model, score = min(model_and_scores, key=lambda item: item[1])
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_and_scores = []

        for current_n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            try:
                hmm_model = GaussianHMM(n_components=current_n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL_self = hmm_model.score(self.X, self.lengths)
                # here we calculate the number of free parameters in the model
                logL_others = []
                for key in self.hwords:
                    if key != self.this_word:
                        another_X, another_lengths = self.hwords[key]
                        logL_others.append(hmm_model.score(another_X,another_lengths))
                DIC_score = logL_self - np.average(logL_others)
                model_and_scores.append([hmm_model, DIC_score])
                if self.verbose:
                    print("model created for {} with {} states and score {} and logL {}".format(self.this_word,
                                                                                                current_n_components,
                                                                                                DIC_score, logL_self))
            except:
                # since model failed it must have infinite BIC score and it will never be selected
                model_and_scores.append([None, float("-inf")])
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, current_n_components))

        best_model, score = max(model_and_scores, key=lambda item: item[1])
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold(n_splits=3)
        model_and_scores = []
        if len(self.sequences)>2:
            for current_n_components in range(self.min_n_components,self.max_n_components+1):
                scores = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_X, train_lenghts = combine_sequences(cv_train_idx,self.sequences)
                    test_X, test_lenghts = combine_sequences(cv_test_idx, self.sequences)

                    try:
                        hmm_model = GaussianHMM(n_components=current_n_components, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(train_X, train_lenghts)
                        scores.append(hmm_model.score(test_X,test_lenghts))
                        if self.verbose:
                            print("model created for {} with {} states".format(self.this_word, current_n_components))

                    except:
                        if self.verbose:
                            print("failure on {} with {} states".format(self.this_word, current_n_components))
                # before returning the model we train on full data
                hmm_model = GaussianHMM(n_components=current_n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                model_and_scores.append([hmm_model,np.mean(scores)])

            best_model, score = max(model_and_scores, key=lambda item: item[1])
        else:
            try:
                best_model = hmm_model = GaussianHMM(n_components=self.n_constant, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            except:
                if self.verbose:
                    print("failure on {} altogether".format(self.this_word))

        return best_model

class Selector_customDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        import heapq

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        model_and_scores = []

        for current_n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            try:
                hmm_model = GaussianHMM(n_components=current_n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL_self = hmm_model.score(self.X, self.lengths)
                # here we calculate the number of free parameters in the model
                logL_others = []
                for key in self.hwords:
                    if key != self.this_word:
                        another_X, another_lengths = self.hwords[key]
                        logL_others.append(hmm_model.score(another_X,another_lengths))
                highest_others = heapq.nlargest(10,logL_others)
                DIC_score = logL_self/np.average(highest_others)
                model_and_scores.append([hmm_model, DIC_score])
                if self.verbose:
                    print("model created for {} with {} states and score {} and logL {}".format(self.this_word,
                                                                                                current_n_components,
                                                                                                DIC_score, logL_self))
            except:
                # since model failed it must have infinite BIC score and it will never be selected
                model_and_scores.append([None, float("inf")])
                if self.verbose:
                    print("failure on {} with {} states".format(self.this_word, current_n_components))

        best_model, score = min(model_and_scores, key=lambda item: item[1])
        return best_model