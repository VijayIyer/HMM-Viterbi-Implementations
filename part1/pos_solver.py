###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import collections
from collections import defaultdict

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "HMM":
            return -999
        elif model == "Complex":
            return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        self.p_words = {}
        self.p_pos = {}
        self.pos_word_pos = {}
        self.total_words = 0
        self.total_pos = 0
        for i, (words, tags) in enumerate(data):
            for word, tag in zip(words,tags):
                if word not in self.p_words:
                    self.p_words[word] = {}
                    self.p_words[word]['total_count'] = 1
                    if tag not in self.p_pos:
                        self.p_pos[tag] = 1
                    else:
                        self.p_pos[tag] += 1
                    if tag not in self.p_words[word].keys():
                        self.p_words[word][tag] = 1
                    # self.p_words[word]['total_count'] = 1
                else:
                    self.p_words[word]['total_count'] += 1
                    if tag not in self.p_words[word].keys():
                        self.p_words[word][tag] = 1
                    else:
                        self.p_words[word][tag] += 1
                self.total_words += 1
                self.total_pos += 1

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        output_pos = []
        for word in sentence:
            output = {k: 0 for k in self.p_pos.keys()}
            if word not in self.p_words:
                for pos in self.p_pos.keys():
                    output[pos] = (1/(self.p_pos[pos]+self.total_words))
            else:
                p_word = self.p_words[word]['total_count']/self.total_words
                for pos in self.p_pos.keys():
                    if pos not in self.p_words[word].keys():
                        output[pos] = 0
                    else:
                        answer = (self.p_words[word][pos]/ self.p_pos[pos]) * (self.p_pos[pos]/self.total_pos)
                        output[pos] = answer
            output_pos.append(max(output, key = lambda t:output[t]))
        return output_pos
    def hmm_viterbi(self, sentence):
        return [random.choice(["noun","verb","adj","pron","det"]) for word in range(len(sentence))]

    def complex_mcmc(self, sentence):
        return [random.choice(["noun","verb","adj","pron","det"]) for word in range(len(sentence))]



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")

