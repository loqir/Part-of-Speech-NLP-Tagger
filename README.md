# Part-of-Speech-NLP-Tagger

This project implements the Viterbi algorithm using training data of tweets and their tags.
A Hidden Markov Model is generated with the tags of each word of the tweets as states and the word as the output.
Maximum Likelihood Estimation is then used to generate the transition probabilities between the states and the output probability from the states (done in projectq4.py)

In hmm.py, implementation of the Viterbi algorithm is done using the transition and output probabilities to predict the most likely sequence of states(tags) given the tweets(words).
An improved version was attempted after analysing the language used in the tweets to better predict the sequences of states, achieving an accuracy of about 80%.
