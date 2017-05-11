import warnings
from asl_data import SinglesData
import numpy as np
import collections


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # TODO implement the recognizer
    probabilities = []
    guesses = []

    for word_id in range(0,len(test_set.get_all_Xlengths())):
    #for word_id in range(0,2):

        current_sequence = test_set.get_item_sequences(word_id)
        current_length = test_set.get_item_Xlengths(word_id)
        probs = {}
        for word, model in models.items():
            try:
                probs[word] = model.score(current_sequence[0], current_length[1])
            except:
                print('failed for word_id {} and word: {}'.format(word_id,word))
                probs[word] = float('-inf')


        guess = max(probs, key=probs.get)
        probabilities.append(probs)
        guesses.append(guess)

    return probabilities, guesses

def recognize_ngram(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    # This import is necessary to be able recognize language model in arpa files
    # could be easily installed using the following command :
    # pip install arpa
    import arpa
    import itertools
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # TODO implement the recognizer
    probabilities = []
    guesses = []
    probabilities_dict = {}
    guesses_dict = {}

    #load the language model
    lm_models = arpa.loadf('lm3_sri.lm')
    lm = lm_models[0]  # ARPA files may contain several models.

    #for word_id in range(0, len(test_set.get_all_Xlengths())):
    #    probabilities_dict[word_id] = 'None'
    #    guesses_dict[word_id] = 'None'

    for video_index in test_set._load_sentence_word_indices():
        word_ids = test_set._load_sentence_word_indices()[video_index]
        video_probs = collections.OrderedDict()
        for word_id in word_ids:

            current_sequence = test_set.get_item_sequences(word_id)
            current_length = test_set.get_item_Xlengths(word_id)
            probs = {}
            for word, model in models.items():
                try:
                    probs[word] = model.score(current_sequence[0], current_length[1])
                except:
                    print('failed for word_id {} and word: {}'.format(word_id,word))
                    probs[word] = float('-inf')

            if len(word_ids)>5:
                top_words = sorted(probs, key=probs.get, reverse=True)[:3]
            elif len(word_ids)==5:
                top_words = sorted(probs, key=probs.get, reverse=True)[:4]
            elif len(word_ids)<5:
                top_words = sorted(probs, key=probs.get, reverse=True)[:6]

            probabilities_dict[word_id] = probs
            video_probs[word_id]={x:probs[x] for x in top_words}

        sentences = list(itertools.product(*video_probs.values()))
        sentences_prob = []


        for sentence_index in range(len(sentences)):
            sentence = sentences[sentence_index]
            visual_prob = 0
            word_index = 0
            for word_id in word_ids:
                word_id_probs = video_probs[word_id]
                visual_prob = visual_prob + word_id_probs[sentence[word_index]]
                word_index = word_index + 1

            sentence_string = ''
            for word in sentence:
                sentence_string = sentence_string + ' ' + word
            try:
                language_prob = lm.log_s(sentence_string.strip())
                alpha = 1
                beta = 25
                sentence_prob = alpha*visual_prob + beta*language_prob
                sentences_prob.append(sentence_prob)
                print(language_prob)
            except:
                print('no language for sor sentence: {}',sentence_string.strip())
                sentences_prob.append(float('-inf'))

        #find the sentence with the highest prob then extract word_ids
        max_sentence = sentences[sentences_prob.index(max(sentences_prob))]

        word_index = 0
        for word_id in word_ids:
            guesses_dict[word_id] = max_sentence[word_index]
            word_index = word_index + 1

    for key in sorted(guesses_dict):
        probabilities.append(probabilities_dict[key])
        guesses.append(guesses_dict[key])

    return probabilities, guesses

