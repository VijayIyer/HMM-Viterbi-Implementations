#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import math
import random
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }



#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!
exemplars = []

#region Training
word_count = {k:0 for k in train_letters.keys()}
file = open(train_txt_fname, 'r')
total_num_lines = 0
for line in file:
    words = line.split()[::2]
    sentence = " ".join(words)
    for character in sentence:
        if character in word_count.keys():
            word_count[character] += 1
    total_num_lines+=1
total_word_count = sum([v for k, v in word_count.items()])
file.seek(0)
#endregion
#region Transition and Initial probabilities
trans_prob = {k:{} for k in word_count.keys()}
for k in trans_prob.keys():
    trans_prob[k] = {key:1 for key in word_count.keys()}
init_prob = {k:1 for k in word_count.keys()}
for line in file:
    words = line.split()[::2]
    sentence = " ".join(words)
    if sentence[0] in init_prob.keys():
        init_prob[sentence[0]] += 1
    for i in range(len(sentence) - 1):
        if sentence[i] in trans_prob.keys() and sentence[i+1] in trans_prob.keys():
            trans_prob[sentence[i]][sentence[i + 1]] += 1
for alphabet in word_count.keys():
    init_prob[alphabet] = init_prob[alphabet]/total_num_lines

for pos1 in word_count.keys():
    transitions_sum = sum([v for k, v in trans_prob[pos1].items()])
    for pos2 in word_count.keys():
        trans_prob[pos1][pos2] = trans_prob[pos1][pos2]/transitions_sum
#endregion
#region Part1 Simple Bayes net
actual_characters = []
m = 0.7
#build up probabilities of each pixel given the word
prob_pixel_word = {k:[[0 for _ in range(len(train_letters[random.choice(list(train_letters))][0]))] for _ in range(len(train_letters[random.choice(list(train_letters))]))] for k in train_letters.keys()}  # hard-coded for now
for letter, pixels in train_letters.items():
    for i, pixel_row in enumerate(pixels):
        for j, pixel in enumerate(pixel_row):
            if pixel == '*':
                prob_pixel_word[letter][i][j] = m
            else:
                prob_pixel_word[letter][i][j] = 1 - m

prob_word_pixel = [{k:0 for k in train_letters.keys()} for _ in range(len(test_letters))]
for ind, letter in enumerate(test_letters):
    for k in train_letters.keys():
        for i,row in enumerate(letter[0]):
            for j,character in enumerate(row):
                if character == '*':
                    prob_word_pixel[ind][k] += math.log(prob_pixel_word[k][i][j])
                else:
                    prob_word_pixel[ind][k] += math.log(1 - prob_pixel_word[k][i][j])
        prob_word_pixel[ind][k] += math.log((1+word_count[k])/total_word_count)
    actual_characters.append(max(prob_word_pixel[ind], key=prob_word_pixel[ind].get))

#endregion
#region Part2 HMM
actual_characters_hmm = []
emission_prob = [{k:1 for k in word_count.keys()} for _ in range(len(test_letters))]
delta = [{} for _ in range(len(test_letters))]
for i in range(len(test_letters)):
    delta[i] = {k: 0 for k in word_count.keys()}
paths = [[None for _ in word_count.keys()] for _ in range(len(test_letters))]
# initializing first column of matrix
initial_pixels = test_letters[0]

for k in word_count.keys():
    for i, row in enumerate(initial_pixels):
        for j, character in enumerate(row):
            if character == '*':
                emission_prob[0][k] *= prob_pixel_word[k][i][j]
            else:
                emission_prob[0][k] *= (1 - prob_pixel_word[k][i][j])
    # emission_prob[0][k] *= math.log((1 + word_count[k]) / total_word_count)
for pos in word_count.keys():
    delta[0][pos] = init_prob[pos]*(emission_prob[0][pos])

for ind in range(1, len(test_letters)):
    for dict_ind, k in enumerate(word_count.keys()):
        for i, row in enumerate(test_letters[ind]):
            for j, character in enumerate(row):
                if character == '*':
                    emission_prob[ind][k] *= prob_pixel_word[k][i][j]
                else:
                    emission_prob[ind][k] *= (1 - prob_pixel_word[k][i][j])

        delta[ind][k] = max([delta[ind-1][i]*trans_prob[i][k] for i in word_count.keys()])*emission_prob[ind][k]
        paths[ind][dict_ind] = np.argmax([delta[ind-1][i]*trans_prob[i][k] for i in word_count.keys()])
    normalized_delta = sum([v for k,v in delta[ind].items()])
    for key in delta[ind].keys():
        delta[ind][key] = delta[ind][key]/normalized_delta
final_path = np.argmax([delta[len(test_letters) - 1][pos] for pos in word_count.keys()])
state_path = [final_path]
# backtracking to get the paths
for t in range(len(test_letters) - 1, 0, -1):
    state_path.append(paths[t][final_path])
    final_path = state_path[-1]
actual_characters_hmm = [list(word_count.keys())[state] for state in state_path]
actual_characters_hmm = actual_characters_hmm[::-1]
#endregion

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
print("\n".join([ r for r in train_letters['a'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
print("\n".join([ r for r in test_letters[2] ]))



# The final two lines of your output should look something like this:
print("Simple: " + "".join(actual_characters))
print("   HMM: " + "".join(actual_characters_hmm))


