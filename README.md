# a3
In each of the problems below, I have implemented HMM - Viterbi algorithm. 
In each case, I try to follow the standard methods of computing MAP of States(**Q**) given a set of Observations (**O**). In each case, for the dynamic programming algorithm, I keep 2 2d Matrices with States as rows and Observations as Columns. The first 2d matrix **delta**, keeps track of the values, **delta[row][col]** represents the probability of the most probable path from starting **Observation 0 to Observation Col having a state [row]**. The second 2d matrix is the paths of each delta[row][col]. At each column, the index at any row is the most proabable previous state for the previous observation - delta[previous-state][col-1]. Having a 2d matrix in both cases in this way, avoids having to do recursive calls, and instead I perform this dynamic programming algorithm in an iterative manner. 
For our computation, the components required are 

## Problem 1 - POS Tagging - 
-  1. Viterbi Algorithm
    - * . Here the problem, is to identify the Parts of speech in sentences of the test corpus, after training on the brown corpus file given.
    - 2. The **States here are the 12 parts of speech tags (Noun, Verb, Adv, Adj, DET, etc.....)** and the observations **(O)** are the words that are given to us in the test sentence. 
    - 3. The transition probabilities are the probabilities of obtaining some State Qi,given the previous State Qj 

## Problem 2 - Mountain Finding - 

## Problem 3 - Image to Text - 
