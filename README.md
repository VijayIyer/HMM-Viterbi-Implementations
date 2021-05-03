# a3
In each of the problems below, I have implemented HMM - Viterbi algorithm. 
In each case, I try to follow the standard methods of computing MAP of States(**Q**) given a set of Observations (**O**). In each case, for the dynamic programming algorithm, I keep 2 2d Matrices with States as rows and Observations as Columns. The first 2d matrix **delta**, keeps track of the values, **delta[row][col]** represents the probability of the most probable path from starting **Observation 0 to Observation Col having a state [row]**. The second 2d matrix is the paths of each delta[row][col]. At each column, the index at any row is the most proabable previous state for the previous observation - delta[previous-state][col-1]. Having a 2d matrix in both cases in this way, avoids having to do recursive calls, and instead I perform this dynamic programming algorithm in an iterative manner. 
For our computation, the components required are 

## Problem 1 - POS Tagging - 
**Simple Bayes or Naive Bayes** - 
 - Here for each word in test sentence, the computation done is for the state that is most probable, regardless of the states ahead or behind. For **implementing** Simple Bayes net model, I keep 2 dictionaries  - 1.Number of occurences of the parts of speech *prob[pos]* 2. Number of occurences of each word for each part of speech. The 2nd dictionary is a dictionary of dictionaries, with each word key containing a dictionary of POS keys. To get the likelihood probability or P(W|POS), I simply look up the count in the dictionary of dictionary like this  - word_pos[word][pos], where pos is the part of the speech. I build the likelihood probabilities or counts in these dictionaries using training data to then later use during test process. P(W|POS) is just built up by keeping the count of number of times word occured for given POS by iterating over words in the training file and incrementing word_pos[word][pos]
  - Then during the testing process, iterating over the sentence, when a word is seen - then for all known POS, we calculate P(POS|Word) by *word_pos[word][pos] * prob[pos]* (the denominator prob[word] is not required as it is a constant and we only require the POS with max value. We then get the POS with the maximum value, and this is the POS for that word. This is done for every word in the test sentence. 
 - **Challenges** - 
  - For **unknown words**, I assigned a probability of *1/count of pos* for **word_pos[word][pos]**. With words there are 2 types of edge cases, 1. when a word itself was never encountered, 2. when a word was never encountered with the particular POS.
  - **Comments and results** - For the test file, bc.test with 200 test sentences, I obtain an accuracy of ***91.76%*** for words correct and ***37.75%*** for number of sentences correct. Since the simple bayes net considers, all states independent given the word, the answer is influenced by count of the POS for the given word. 

**Viterbi Algorithm** -  
 - Here the problem, is to identify the Parts of speech in sentences of the test corpus, after training on the brown corpus file given.
 - The **States here are the 12 parts of speech tags (Noun, Verb, Adv, Adj, DET, etc.....)** and the observations **(O)** are the words that are given to us in the test sentence. 
 - The transition probabilities are the probabilities of obtaining some State Qi,given the previous State Qj. 
 - **Implementation** - I go over each training sentence just as for the simple bayes model. There are 2 dictionaries  - 1. **init_prob** dictionary holding the initial probability of a sentence starting with a given POS tag. For this, I increment the value of the dictionary key which occurs as the first tag in the training sentence. 2. **trans_prob** - a dictionary of dictionaries, with rows and columns both as pos tags. In this case, since there are 12 POS, our matrix is of size 12 by 12. I go over each word with its POS in training text, and then consider the next word to increment trans_prob[pos1][pos2] by 1 which signifies the transition of POS tag1 to POS tag2. The **emission probability** of a word is just the dictionary calculated in the previous Simple Bayes model **word_pos[word]|[pos]**. As discussed above, the delta matrix holds the running values required in the Viterbi MAP paths. The Steps in the dynamic programming solution are as follows - 1. delta[pos][0] is updated for every POS using **init_prob[pos] * word_pos[word][pos]**. Once we have the first column initialized, that is the delta values for the first word in the sentence(meaning the probabilites of the word being any of the states), we iterate from the 2nd word to the last word. Every next delta column value is computed as follows - delta[col_pos][col] = max(delta[previous_col_pos][col-1] for every pos)* word_pos[word(observation at col)][col_pos]. This value is the maximum probability of the most likely State Sequences from the start till current word for the current pos being considered. I also update path for each POS at every word by taking np.argmax(delta[previous_col_pos][col-1] for every pos). At every column, the value at a given row shows the most likely previous state. Once this is complete for all the words, we take np.argmax(..) for the delta values in the last column again and record the POS. Then using the value at this POS row in the last column, we obtain the most likely POS in the previous Column, and so on, to append these values in a list return that list
 - **Results** - 
  - Obtained accuracy with bc.test file was **94.35%** for words correct, and **50.60** for the sentences correct. 
  - A lot more accuracy was expected for the sentences correct, as HMM models the transition probabilities and is a richer model

**Complex Bayes Net(Gibbs Sampling)** - 
 - Here, I am using np.random.choice for sampling POS tags from the list of POS by building a probability distribution for each key at that index of the sentence. 
 - I start with an initial list of POS tags. Then loop over all the indexes. For every index, I calculate the probability distribution of all POS tags possible at that index, with other index values fixed. In the training step, I build a dictionary of dictionaries with keys as tuples, it is in the form **dict[word]|[(pos1,pos2)]**, where word is the observation at the index about to be sampled, pos1 is the POS tag of previous index, since as per our bayes net model, word is dependent on both current POS tag and previous word's POS tag. The probability distribution from which sampling is to be done is a list with probabilities of each POS key. It is calculated as  - 
 **P(W|S1,S2)* P(S2|S1) * P(S1)**, where the first term is represented by our **dict[word]|[(pos1,pos2)]** form object that we have computed from the training corpus. P(S2|S1) is simply the transition probabilities calculated before, while P(S1) is the probability of POS tag itself, independent of other tags. At any point in the observation sequence, given we know POS1, POS2 is conditionally independent of all previous POS tags before POS1. 
 - **Implementation** - As mentioned above, np.random.choice is used for sampling a value from the dictionary of POS tags, with one of the arguments passed as the probability distribution of each tag. I also defined 2 variables, k, the number of iterations, and break_in, the number of samples after which obtained samples are to be counted. For counting the samples I use a dictionary with the keys the tuple of POS tags. After the break_in, I increment the values of this dictionary as per the obtained sample. 
 - **Results and Challenges** - 
  - For the Complex Bayes Net, the words correct is coming out to be around **85%**, while sentences correct is coming to be under **10%**
  - I keep k = 1000 and break_in  = 500, thus for every word in the sentence, 500 samples are created and the sequence with max probability is returned. Ideally, break_in of 1000 and K= 1000 could work better, however, its taking a second per test sample, thus around 30 minutes for the entire test set. 
 
**Posterior Probabilities** - 
 - For the Simple Bayes net, the Log of posterior probabilitiy is calculated as the product of P(S|W) for each S and W. This is because as per our model in fig 1b of assignment text, all POS tags are independent of each other
 - For the HMM model, the posterior probability is calculated as the product of emission probabilities for every observation(word) and transition probabilities for every transition (pos1 -> pos2).
  - For the complex model, the formula I used is  - **P(wn|Sn,Sn-1)* P(Sn-1|Sn)* P(wn-1|Sn-1,Sn-2) * P(Sn-2|Sn-1).. * P(S1).**, that is, I use the same formula used for creating probability distribution for a POS tag before, and take product of all such pairs, with final multiplication being with P(S1). 

## Problem 2 - Mountain Finding - 
I have used the same approach as in Viterbi Algorithm for problem 1 and 3
 - States - Pixel location per Column which can take any value between 0 and height of image, Observations -  all the pixels in One column
 - Transition probabilities -a 2d matrix denoting the probability of going from a pixel1 in col1 to pixel2 in next column. Here transition probabilities are dependent only on the pixel location. So I am penalising transition from any pixel to those pixels in the next column which are more than 1 pixel away. The formula I am using is ***if pixel row is same or only difference of 1 for both columns, then transition probability = 1, else, 1/abs(pixel1 - pixel2)^2, then normalizing over all the pixel values for a row***. The power of the denominator decides whether more smoothing is required. It will prevent bigger jumps in the identified pixel locations of the ridgeline. The other option is a gaussian distribution, but I am unable to resolve the issues with 0 since, for large differences the exponent becomes huge. For known co-ordinate problem, at the column location given, I set all other values except those going to the row location to zero, this the location gets picked.  
  - I am also normalizing the delta values at each column in this problem, as values get very small for huge length sequences over the width of the image.
  - Emission probabilities(probability of pixel values being what they are given some pixel in the column is the ridgeline location) - product of gradient strengh at pixel location * (1 - gradient strength at other locations). This is to ensure, that high gradient strengths are given higher probabilities.
 
 - For the simple bayes net I am just taking the pixel location with the maximum gradient value. However, the better choice would be some function which penalizes variance of gradient strengh above the pxiel location - this however didnt work in practice, and the pixel being picked is still the one with highest gradient strength irrespective of above values
 - **Results and Challenges** - The emission probabilites were hard to decide, and none of the functions I decided were able to pick the right location when some other location had higher gradient strength. Even with known co-ordinates, 
 

## Problem 3 - Image to Text - 
- 1.**Naive Bayes Net** - Similar to problem 1, I try to compute P(State|Observation) for each observation independently. Here the states are the letters and the observations are the 2d grid of pixels generated from the method which reads the image. To model this I keep a dictionary of 2d arrays, with every entry in the array, the probability of the pixel being an asterisk. The reason this can be done is because( as per hint given in the assignment document), since the pixel can only take 2 values, it can be stored with 1 value m with the complementary value being 1 - m. I go over each letter pixel configuration, the training sentence, courier-train.png, and have a dictionary in place for each alphabet/letter. When I see a asterisk, for a known letter in the trainining, I store value m at that location. For eg, if pixel (1, 1) is asterisk for the training letter image of A, then I store m = (some value between 0 and 1) and if (1, 1) is blank, then 1-m is assigned to that pixel location. When going over the test letters, similar to the problem 1 , for each letter, I compute the P(Observation|Letter) by taking product of -> P(pixel|Letter) for each pixel. When an asterisk is encountered then I multiply the running count variable(this is the running count of the product of P(pixel|Letter)) by pixel_location_value (this value can be m or 1-m, depending on if the pixel value in the test image is same as in train image for that letter or not). Thus, the pixel location value is a measure of agreement with a letter. If many pixel values are same as for any of the letters, then the running product count will be high and this will be the key with maximum probability. This is then done for every letter image in the test image
- 2.**Viterbi Algorithm** - This algorithm is implemented the same way as in problem 1 with delta 2d matrix being built up over the observation sequence. The few differences are, in this case the **transition probabilities** are built up over every 2 letters instead of POS tags in the train document. I used the same bc.train file of problem 1 with some preprocessing after reading the file. I removed the POS labels which are every 2nd word in the bc.train file. The **emission probabilities** are calculated the same way as product of P(pixel|letter) for every pixel as in the Naive Bayes approach above. The **initial probabilities** are the counts of letters occuring as the first character divided by the number of sentences. 
- 3.**Challenges and Results** -The value of m heavily influences how a letter is identified in a noisy image or clean image, this value is hard to predict. For the naive bayes approach, the number of occurences of the letter ' ', heavily influenced the noisy images. 
- 4.In both cases, for the test sentences images, there are particular words which seem to always be misclassified. The Viterbi solution in general seems to get more characters correct. The common letters which are misidentified are 'd' identified as 'c', 'i' identified as '1', a very noisy image is just shown as some _ and ' '. Thus, performance on the noisy image is poor. The emission probability computation is the major challenge in this problem, since, in this case, there are no training images, and taking each pixel seperately means for each State there are a set of observation instead of 1 observation. 
  
