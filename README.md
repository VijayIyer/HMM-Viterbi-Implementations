# a3
In each of the problems below, I have implemented HMM - Viterbi algorithm. 
In each case, I try to follow the standard methods of computing MAP of States(**Q**) given a set of Observations (**O**). In each case, for the dynamic programming algorithm, I keep 2 2d Matrices with States as rows and Observations as Columns. The first 2d matrix **delta**, keeps track of the values, **delta[row][col]** represents the probability of the most probable path from starting **Observation 0 to Observation Col having a state [row]**. The second 2d matrix is the paths of each delta[row][col]. At each column, the index at any row is the most proabable previous state for the previous observation - delta[previous-state][col-1]. Having a 2d matrix in both cases in this way, avoids having to do recursive calls, and instead I perform this dynamic programming algorithm in an iterative manner. 
For our computation, the components required are 

## Problem 1 - POS Tagging - 
###. **Simple Bayes or Naive Bayes** - 
 - Here for each word in test sentence, the computation done is for the state that is most probable, regardless of the states ahead or behind. For **implementing** Simple Bayes net model, I keep 2 dictionaries  - 1.Number of occurences of the parts of speech *prob[pos]* 2. Number of occurences of each word for each part of speech. The 2nd dictionary is a dictionary of dictionaries, with each word key containing a dictionary of POS keys. To get the likelihood probability or P(W|POS), I simply look up the count in the dictionary of dictionary like this  - word_pos[word][pos], where pos is the part of the speech. I build the likelihood probabilities or counts in these dictionaries using training data to then later use during test process. P(W|POS) is just built up by keeping the count of number of times word occured for given POS by iterating over words in the training file and incrementing word_pos[word][pos]
  - Then during the testing process, iterating over the sentence, when a word is seen - then for all known POS, we calculate P(POS|Word) by *word_pos[word][pos] * prob[pos]* (the denominator prob[word] is not required as it is a constant and we only require the POS with max value. We then get the POS with the maximum value, and this is the POS for that word. This is done for every word in the test sentence. 
  - ** Challenges** - 
   - For **unknown words**, I assigned a probability of *1/count of pos* for **word_pos[word][pos]**. With words there are 2 types of edge cases, 1. when a word itself was never encountered, 2. when a word was never encountered with the particular POS.
  - **Comments and results** - For the test file, bc.test with 200 test sentences, I obtain an accuracy of ***91.76%*** for words correct and ***37.75%*** for number of sentences correct. Since the simple bayes net considers, all states independent given the word, the answer is influenced by count of the POS for the given word. 
###. Viterbi Algorithm
    - Here the problem, is to identify the Parts of speech in sentences of the test corpus, after training on the brown corpus file given.
    - The **States here are the 12 parts of speech tags (Noun, Verb, Adv, Adj, DET, etc.....)** and the observations **(O)** are the words that are given to us in the test sentence. 
    - The transition probabilities are the probabilities of obtaining some State Qi,given the previous State Qj. 
    - **Implementation** - 

## Problem 2 - Mountain Finding - 

## Problem 3 - Image to Text - 
