###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import numpy as np
import collections
from collections import defaultdict
import itertools

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        if model == "Simple":
            ans = 0
            temp_dict = {}              # a temporary dict to hold prob of words and pos not yet seen
            for word, pos in zip(sentence, label):
                if word not in self.p_words.keys():
                    prob_word = 1/self.total_words
                    if pos not in self.p_pos.keys():
                        prob_pos = 1 / self.total_pos
                        prob_pos_word = 1 / (self.total_pos*self.total_words)
                    else:
                        prob_pos = self.p_pos[pos]/self.total_pos
                        prob_pos_word = 1 / self.p_pos[pos]
                else:
                    prob_word = self.p_words[word]['total_count']/self.total_words
                    if pos not in self.p_pos.keys():
                        prob_pos = 1/self.total_pos
                        prob_pos_word = self.p_words[word][pos] / self.p_pos[pos]

                    else:
                        prob_pos = self.p_pos[pos]/self.total_pos
                        if pos not in self.p_words[word].keys():
                            prob_pos_word = 1 / self.p_pos[pos]
                        else:
                            prob_pos_word = self.p_words[word][pos] / self.p_pos[pos]

                ans += math.log((prob_pos_word)*(prob_pos)*(1/prob_word))
            return ans
        elif model == "HMM":
            ans = 0
            # emission_prob = [0 for _ in range(len(sentence))]
            for word, pos in zip(sentence, label):
                if word not in self.p_words.keys():
                    prob_word = 1 / self.total_words
                    if pos not in self.p_pos.keys():
                        prob_pos = 1 / self.total_pos
                        prob_pos_word = 1 / (self.total_pos * self.total_words)
                    else:
                        prob_pos = self.p_pos[pos] / self.total_pos
                        prob_pos_word = 1 / self.p_pos[pos]
                else:
                    prob_word = self.p_words[word]['total_count'] / self.total_words
                    if pos not in self.p_pos.keys():
                        prob_pos = 1 / self.total_pos
                        prob_pos_word = self.p_words[word][pos] / self.p_pos[pos]

                    else:
                        prob_pos = self.p_pos[pos] / self.total_pos
                        if pos not in self.p_words[word].keys():
                            prob_pos_word = 1 / self.p_pos[pos]
                        else:
                            prob_pos_word = self.p_words[word][pos] / self.p_pos[pos]
                ans += math.log(prob_pos_word)

            for i in range(len(label)-1):
                if i == 1:
                    ans += math.log(self.init_prob[label[i - 1]])
                else:
                    ans += math.log(self.trans_prob[label[i-1]][label[i]])

            return ans
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
                    if tag not in self.p_pos:
                        self.p_pos[tag] = 1
                    else:
                        self.p_pos[tag]+=1
                    if tag not in self.p_words[word].keys():
                        self.p_words[word][tag] = 1
                    else:
                        self.p_words[word][tag] += 1
                self.total_words += 1
                self.total_pos += 1

        # Transition probabilities
        self.trans_prob = {k:{} for k in self.p_pos.keys()}
        for k in self.trans_prob.keys():
            self.trans_prob[k] = {key:1 for key in self.p_pos.keys()}
        self.init_prob = {k:1 for k in self.p_pos.keys()}
        for sentence in data:
            self.init_prob[sentence[1][0]] += 1
            for i in range(len(sentence[1])-1):
                self.trans_prob[sentence[1][i]][sentence[1][i+1]] += 1

        for pos in self.p_pos.keys():
            self.init_prob[pos] = self.init_prob[pos]/len(data)

        for pos1 in self.p_pos.keys():
            transitions_sum = sum([v for k, v in self.trans_prob[pos1].items()])
            for pos2 in self.p_pos.keys():
                self.trans_prob[pos1][pos2] = self.trans_prob[pos1][pos2]/transitions_sum

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
        delta = [{} for _ in range(len(sentence))]
        for i in range(len(sentence)):
            delta[i] = {k:0 for k in self.p_pos.keys()}
        paths = [[None for _ in self.p_pos.keys()] for _ in range(len(sentence))]

        # initializing first column of matrix
        for i, pos in enumerate(self.p_pos.keys()):
            if sentence[0] not in self.p_words.keys():
                delta[0][pos] = self.init_prob[pos]*(1/self.p_pos[pos])

            else:
                if pos not in self.p_words[sentence[0]].keys():
                    delta[0][pos] = 0
                else:
                    delta[0][pos] = self.init_prob[pos]*(self.p_words[sentence[0]][pos]/self.p_pos[pos])

        # recursive solution
        for t in range(1, len(sentence)):
            for i, current_pos in enumerate(self.p_pos.keys()):
                if sentence[t] not in self.p_words.keys():
                    prob_word_pos = 1/self.p_pos[current_pos]
                elif current_pos in self.p_words[sentence[t]].keys():
                    prob_word_pos = self.p_words[sentence[t]][current_pos]/self.p_pos[current_pos]
                else:
                    prob_word_pos = 0
                delta[t][current_pos] = max([delta[t - 1][pos] * self.trans_prob[pos][current_pos] for pos in self.p_pos.keys()])*prob_word_pos
                paths[t][i] = np.argmax([delta[t - 1][pos] * self.trans_prob[pos][current_pos] for pos in self.p_pos.keys()])
        final_prob = max([delta[len(sentence)-1][pos] for pos in self.p_pos.keys()])
        final_path = np.argmax([delta[len(sentence)-1][pos] for pos in self.p_pos.keys()])
        state_path = [final_path]
        # backtracking to get the paths
        for t in range(len(sentence)-1, 0, -1):
            state_path.append(paths[t][final_path])
            final_path = state_path[-1]
        pos_path = [list(self.p_pos.keys())[state] for state in state_path]
        return pos_path[::-1]

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

