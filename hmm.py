# Implement the six functions below
import ast
delta = 0.1
delta_transition = 0.1
smooth_dic = {}
words_set = set()
with open("twitter_train.txt", "r", encoding="utf8") as input:
    for line in input:
        both = line.split()
        if len(both) > 0:
            word = both[0].lower()
            words_set.add(word)
            tag = both[1]
            if tag in smooth_dic:
                if word not in smooth_dic[tag]:
                    smooth_dic[tag][word] = 1
                else:
                    smooth_dic[tag][word] += 1
            else:
                smooth_dic[tag] = {word: 1}
    num_words = len(words_set)

    for tagkey in smooth_dic:
        countyj = sum(smooth_dic[tagkey].values())
        smooth_dic[tagkey]["SPECIFIC_TAG_COUNT"] = countyj
        for word in smooth_dic[tagkey]:
            if word != "SPECIFIC_TAG_COUNT":
                count = smooth_dic[tagkey][word]
                count += delta
                fraction = count / (countyj + delta * (num_words + 1))
                smooth_dic[tagkey][word] = fraction


with open("naive_output_probs.txt", 'w', encoding="utf8") as outputSMOOTH:
    outputSMOOTH.write(str(smooth_dic))


def get_num_words(in_output_probs_filename):
    with open(in_output_probs_filename, "r", encoding="utf8") as dic:
        words_set = set()
        dic = dic.read()
        dic = ast.literal_eval(dic)
        for tagkey in dic:
            for word in dic[tagkey]:
                words_set.add(word)
    return len(words_set)


def get_word_set(in_output_probs_filename):
    with open(in_output_probs_filename, "r", encoding="utf8") as dic:
        words_set = set()
        dic = dic.read()
        dic = ast.literal_eval(dic)
        for tagkey in dic:
            for word in dic[tagkey]:
                words_set.add(word)
        return words_set


def get_word_count(in_train_filename, word):
    with open(in_train_filename, "r", encoding="utf8") as input:
        word_count = 0
        for line in input:
            both = line.split()
            if len(both) > 0:
                appeared_word = both[0].lower()
                if word == appeared_word:
                    word_count += 1
    return word_count


def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    with open(in_output_probs_filename, "r", encoding="utf8") as dic:
        num_words = get_num_words(in_output_probs_filename)
        dic = dic.read()
        dic = ast.literal_eval(dic)
        with open(in_test_filename, "r", encoding="utf8") as testwords:
            with open(out_prediction_filename, 'w', encoding="utf8") as output:
                for word in testwords:
                    if word == "\n":
                        output.write("\n")
                    else:
                        word = word.strip().lower()
                        probability_dic_for_word = {}
                        for key in dic:
                            countyj = dic[key]["SPECIFIC_TAG_COUNT"]
                            if word not in dic[key]:
                                probability_dic_for_word[key] = delta / \
                                    (countyj + delta * (num_words + 1))
                            else:
                                probability_dic_for_word[key] = dic[key][word]
                        best_tag = max(probability_dic_for_word,
                                       key=probability_dic_for_word.get)
                        output.write(best_tag + "\n")


naive_predict("naive_output_probs.txt",
              "twitter_dev_no_tag.txt", "naive_predictions.txt")


def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    with open(in_output_probs_filename, "r", encoding="utf8") as dic:
        num_words = get_num_words(in_output_probs_filename)
        dic = dic.read()
        dic = ast.literal_eval(dic)
        with open(in_test_filename, "r", encoding="utf8") as testwords:
            with open(out_prediction_filename, 'w', encoding="utf8") as output:
                for word in testwords:
                    if word == "\n":
                        output.write("\n")
                    else:
                        word = word.strip().lower()
                        if word.startswith("@"):
                            output.write("@" + "\n")
                            continue

                        word_count = get_word_count(in_train_filename, word)
                        probability_dic_for_word = {}
                        for key in dic:
                            countyj = dic[key]["SPECIFIC_TAG_COUNT"]
                            if word not in dic[key]:
                                fraction = delta / \
                                    (countyj + delta * (num_words + 1))
                                fraction *= (countyj + delta * (num_words + 1))
                                fraction /= (word_count + delta)
                            else:
                                fraction = dic[key][word]
                                fraction *= (countyj + delta * (num_words + 1))
                                fraction /= (word_count + delta)
                            probability_dic_for_word[key] = fraction
                        best_tag = max(probability_dic_for_word,
                                       key=probability_dic_for_word.get)
                        # if word not in get_word_set(in_output_probs_filename):
                        #     print(best_tag)
                        output.write(best_tag + "\n")


naive_predict2("naive_output_probs.txt", "twitter_train.txt",
               "twitter_dev_no_tag.txt", "naive_predictions2.txt")


def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename, in_train_filename):
    with open(in_output_probs_filename, "r", encoding="utf8") as dic:
        num_words = get_num_words(in_output_probs_filename)
        dic = dic.read()
        dic = ast.literal_eval(dic)
        with open(in_test_filename, "r", encoding="utf8") as testwords:
            with open(in_trans_probs_filename, "r", encoding="utf8") as trans_dic:
                trans_dic = trans_dic.read()
                trans_dic = ast.literal_eval(trans_dic)
                with open(in_tags_filename, "r", encoding="utf8") as tags:
                    all_tags = []
                    for tag in tags:
                        all_tags.append(tag.strip())
                    with open(out_predictions_filename, "w", encoding="utf8") as output:
                        num_tags = 0
                        for tag in all_tags:
                            num_tags += 1

                        # initialising empty transition matrix and backpointer matrix
                        trans_matrix = []
                        bp_matrix = []

                        # test_words is to keep all the words in the current tweet to perform an iteration of viterbi and HMM model over each tweet
                        test_words = []
                        for word in testwords:
                            if word == "\n":  # once encountered an empty line, use all the words in test_words to execute one iteration of viterbi
                                curr_word = test_words[0].lower()
                                # using twitter train data to get word count used for smoothing
                                word_count = get_word_count(
                                    in_train_filename, word)
                                pi_1_tags = []  # this is the first layer of the transition matrix made up of function of initial and emission probabilities
                                for tag in all_tags:  # iterate through all the tags to calculate each probability
                                    # tag count used for smoothing
                                    countyj = dic[tag]["SPECIFIC_TAG_COUNT"]
                                    # if there does not exist emission probability of the tag for this current word
                                    if curr_word not in dic[tag]:
                                        # smoothing
                                        emission_prob = delta / \
                                            (countyj + delta * (num_words + 1))
                                        emission_prob *= (countyj +
                                                          delta * (num_words + 1))
                                        emission_prob /= (word_count + delta)
                                    else:
                                        # smoothing
                                        emission_prob = dic[tag][curr_word]
                                        emission_prob *= (countyj +
                                                          delta * (num_words + 1))
                                        emission_prob /= (word_count + delta)
                                    # initial probability of tag not found, do smoothing
                                    if ('initial', str(tag)) not in trans_dic:
                                        initial_prob = delta_transition / \
                                            (trans_dic["total_tweets_count"] + delta_transition * (
                                                trans_dic["unique_tag_count"] + 1))
                                    else:
                                        initial_prob = trans_dic[(
                                            'initial', str(tag))]
                                    # multiply initial probability by emission probability
                                    pi_1_tag = initial_prob * emission_prob
                                    pi_1_tags.append(pi_1_tag)
                                # first layer of the transition matrix containing all the highest and only probability of going from start to that state and emitting word
                                trans_matrix.append(pi_1_tags)
                                # taking into account the case where the tweet is only of 1 word, so cannot run the viterbi
                                if len(test_words) == 1:
                                    # extracting the first layer of transition matrix
                                    prev_pi_trans_prob_list = trans_matrix[-1]
                                    compare_prob = []  # using this to compare all the probabilities
                                    for prob in prev_pi_trans_prob_list:  # going to multiply by termination probability since viterbi ends here
                                        index = prev_pi_trans_prob_list.index(
                                            prob)
                                        tag = all_tags[index]

                                        # termination probability of tag not found, do smoothing
                                        if ('termination', str(tag)) not in trans_dic:
                                            termination_prob = delta_transition / \
                                                (trans_dic["total_tweets_count"] + delta_transition * (
                                                    trans_dic["unique_tag_count"] + 1))
                                        else:
                                            termination_prob = trans_dic[(
                                                'termination', str(tag))]

                                        curr_prob = prob * termination_prob  # multiply by termination probability
                                        # comparing all the probabilities to find the most likely tag
                                        compare_prob.append(curr_prob)
                                    pi_i_tag = max(compare_prob)
                                    index = compare_prob.index(pi_i_tag)
                                    output_tag = all_tags[index]
                                    # if test_words[0].startswith("@"):
                                    #     output.write("@" + "\n")
                                    # else:
                                    # predicting tag
                                    output.write(output_tag + "\n")
                                else:
                                    # if tweet is more than 1 word, continue viterbi
                                    for i in range(1, len(test_words)):
                                        curr_word = test_words[i]
                                        pi_i_tags = []
                                        backpointers = []
                                        for tag in all_tags:
                                            # tag count for smoothing
                                            countyj = dic[tag]["SPECIFIC_TAG_COUNT"]
                                            # get the current last layer of transition matrix
                                            prev_pi_trans_prob_list = trans_matrix[-1]
                                            compare_prob = []  # comparing all the previous probabilities
                                            for prob in prev_pi_trans_prob_list:
                                                prev_tag_index = prev_pi_trans_prob_list.index(
                                                    prob)
                                                prev_tag = all_tags[prev_tag_index]
                                                # transition probability of tags not found, do smoothing
                                                if (prev_tag, tag) not in trans_dic:
                                                    transition_prob = delta_transition / (trans_dic["flattened_matrix"].count(prev_tag) +
                                                                                          delta_transition * (trans_dic["unique_tag_count"] + 1))
                                                else:
                                                    transition_prob = trans_dic[(
                                                        prev_tag, tag)]
                                                # emission probability of tag not found for this given word, do smoothing
                                                if curr_word not in dic[tag]:
                                                    emission_prob = delta / \
                                                        (countyj + delta *
                                                         (num_words + 1))
                                                else:
                                                    emission_prob = dic[tag][curr_word]
                                                # if this is the last word, need to multiply by termination probability
                                                if i == len(test_words) - 1:
                                                    # termination probability of tag not found, do smoothing
                                                    if ('termination', str(tag)) not in trans_dic:
                                                        termination_prob = delta_transition / \
                                                            (trans_dic["total_tweets_count"] + delta_transition * (
                                                                trans_dic["unique_tag_count"] + 1))
                                                    else:
                                                        termination_prob = trans_dic[(
                                                            'termination', str(tag))]
                                                    curr_prob = prob * transition_prob * emission_prob * termination_prob
                                                else:
                                                    curr_prob = prob * transition_prob * emission_prob
                                                compare_prob.append(curr_prob)
                                            # finding the max probability to get to this current state and emitting this word
                                            pi_i_tag = max(compare_prob)
                                            index = compare_prob.index(
                                                pi_i_tag)
                                            # locating the previous tag that resulted in the highest probability
                                            backpointer = all_tags[index]
                                            # building this layer of transition matrix
                                            pi_i_tags.append(pi_i_tag)
                                            # building this layer of backpointer matrix
                                            backpointers.append(backpointer)

                                        # building the transition matrix
                                        trans_matrix.append(pi_i_tags)
                                        # building the backpointer matrix
                                        bp_matrix.append(backpointers)

                                    output_tags = []  # once transition and backpointer matrix have been built, generate outputs
                                    last_list = trans_matrix.pop()  # getting the last layer of transition matrix
                                    last_bp_list = bp_matrix.pop()  # getting the last layer of backpointer rmatrix
                                    # finding the maximum probability
                                    last_max_prob = max(last_list)
                                    max_index = last_list.index(last_max_prob)
                                    # finding the best tag with highest probability
                                    last_best_tag = all_tags[max_index]
                                    output_tags.append(last_best_tag)
                                    # locating the previous tag from the last layer of the backpointer matrix
                                    next_tag = last_bp_list[max_index]
                                    # storing the outputs
                                    output_tags.append(next_tag)

                                    while bp_matrix:  # find the backpointers until the backpointer matrix is empty
                                        # finding the index of the tag that was backpointed to
                                        next_index = all_tags.index(next_tag)
                                        prev_bp_list = bp_matrix.pop()  # getting the last layer of backpointer matrix
                                        # using the index to find the next backpointed tag
                                        next_tag = prev_bp_list[next_index]
                                        # adding the tag to list of outputs
                                        output_tags.append(next_tag)

                                    # since tags were backpointed, reverse the list to get the correct direction
                                    output_tags.reverse()
                                    # indexes = []
                                    # for i in range(len(test_words)):
                                    #     if test_words[i].startswith("@"):
                                    #         indexes.append(i)
                                    # for index in indexes:
                                    #     output_tags[index] = "@"

                                    while output_tags:  # writing the output to the output file
                                        output.write(output_tags.pop(0) + "\n")
                                output.write("\n")
                                trans_matrix = []  # this marks the end of the viterbi algorithm, clear the transition matrix, backpointer matrix and list of test words for the next viterbi iteration
                                bp_matrix = []
                                test_words = []
                                continue  # go on to the next tweet

                            else:  # there is no empty line encountered, add the current word to the list of test words
                                test_words.append(word.strip().lower())


viterbi_predict("twitter_tags.txt", "trans_probs.txt", "output_probs.txt", "twitter_dev_no_tag.txt",
                "viterbi_predictions.txt", "twitter_train.txt")


# for question 5, we made a few changes to the viterbi algorithm. Apart from observing patterns in the tag generation of words like how a word starting with @ will generate tag @
# a word starting with # will generate tag #, we also utilised linguistical probabilities of words to improve our algorithm. For example, we identified words involving punctuations
# and correspondingly predicted the tag ",". Another example is how we identified objects like i you me will generate a tag O. As such, we furthered identified these kinds of patterns
# and linguistical properties of the words emittied to generate better tags after running our viterbi algorithm. However, we are aware that the viterbi algorithm was generated from
# the initial unmodified tags so this method in a way tweaks with the backpointing algorithm so it may not make the most sense. Another method we have tried is to use differing deltas
# for the smoothing of emission and transition probabilities since they are smoothed for different purposes and under different circumstances. As such, a different value of delta for
# the smoothing of different kind of probabilities may prove useful. However, we discovered that the best combination of delta was for both to take the value of 0.1.

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename, in_train_filename):
    with open(in_output_probs_filename, "r", encoding="utf8") as dic:
        num_words = get_num_words(in_output_probs_filename)
        dic = dic.read()
        dic = ast.literal_eval(dic)
        with open(in_test_filename, "r", encoding="utf8") as testwords:
            with open(in_trans_probs_filename, "r", encoding="utf8") as trans_dic:
                trans_dic = trans_dic.read()
                trans_dic = ast.literal_eval(trans_dic)
                with open(in_tags_filename, "r", encoding="utf8") as tags:
                    all_tags = []
                    for tag in tags:
                        all_tags.append(tag.strip())
                    with open(out_predictions_filename, "w", encoding="utf8") as output:
                        num_tags = 0
                        for tag in all_tags:
                            num_tags += 1

                        # initialising empty transition matrix and backpointer matrix
                        trans_matrix = []
                        bp_matrix = []

                        # test_words is to keep all the words in the current tweet to perform an iteration of viterbi and HMM model over each tweet
                        test_words = []
                        for word in testwords:
                            if word == "\n":  # once encountered an empty line, use all the words in test_words to execute one iteration of viterbi
                                curr_word = test_words[0].lower()
                                # using twitter train data to get word count used for smoothing
                                word_count = get_word_count(
                                    in_train_filename, word)
                                pi_1_tags = []  # this is the first layer of the transition matrix made up of function of initial and emission probabilities
                                for tag in all_tags:  # iterate through all the tags to calculate each probability
                                    # tag count used for smoothing
                                    countyj = dic[tag]["SPECIFIC_TAG_COUNT"]
                                    # if there does not exist emission probability of the tag for this current word
                                    if curr_word not in dic[tag]:
                                        # smoothing
                                        emission_prob = delta / \
                                            (countyj + delta * (num_words + 1))
                                        emission_prob *= (countyj +
                                                          delta * (num_words + 1))
                                        emission_prob /= (word_count + delta)
                                    else:
                                        # smoothing
                                        emission_prob = dic[tag][curr_word]
                                        emission_prob *= (countyj +
                                                          delta * (num_words + 1))
                                        emission_prob /= (word_count + delta)
                                    # initial probability of tag not found, do smoothing
                                    if ('initial', str(tag)) not in trans_dic:
                                        initial_prob = delta_transition / \
                                            (trans_dic["total_tweets_count"] + delta_transition * (
                                                trans_dic["unique_tag_count"] + 1))
                                    else:
                                        initial_prob = trans_dic[(
                                            'initial', str(tag))]
                                    # multiply initial probability by emission probability
                                    pi_1_tag = initial_prob * emission_prob
                                    pi_1_tags.append(pi_1_tag)
                                # first layer of the transition matrix containing all the highest and only probability of going from start to that state and emitting word
                                trans_matrix.append(pi_1_tags)
                                # taking into account the case where the tweet is only of 1 word, so cannot run the viterbi
                                if len(test_words) == 1:
                                    # extracting the first layer of transition matrix
                                    prev_pi_trans_prob_list = trans_matrix[-1]
                                    compare_prob = []  # using this to compare all the probabilities
                                    for prob in prev_pi_trans_prob_list:  # going to multiply by termination probability since viterbi ends here
                                        index = prev_pi_trans_prob_list.index(
                                            prob)
                                        tag = all_tags[index]

                                        # termination probability of tag not found, do smoothing
                                        if ('termination', str(tag)) not in trans_dic:
                                            termination_prob = delta_transition / \
                                                (trans_dic["total_tweets_count"] + delta_transition * (
                                                    trans_dic["unique_tag_count"] + 1))
                                        else:
                                            termination_prob = trans_dic[(
                                                'termination', str(tag))]

                                        curr_prob = prob * termination_prob  # multiply by termination probability
                                        # comparing all the probabilities to find the most likely tag
                                        compare_prob.append(curr_prob)
                                    pi_i_tag = max(compare_prob)
                                    index = compare_prob.index(pi_i_tag)
                                    output_tag = all_tags[index]
                                    # predicting tag @ if word starts with @
                                    if test_words[0].startswith("@"):
                                        output.write("@" + "\n")
                                    else:
                                        output.write(output_tag + "\n")
                                else:
                                    # if tweet is more than 1 word, continue viterbi
                                    for i in range(1, len(test_words)):
                                        curr_word = test_words[i]
                                        pi_i_tags = []
                                        backpointers = []
                                        for tag in all_tags:
                                            # tag count for smoothing
                                            countyj = dic[tag]["SPECIFIC_TAG_COUNT"]
                                            # get the current last layer of transition matrix
                                            prev_pi_trans_prob_list = trans_matrix[-1]
                                            compare_prob = []  # comparing all the previous probabilities
                                            for prob in prev_pi_trans_prob_list:
                                                prev_tag_index = prev_pi_trans_prob_list.index(
                                                    prob)
                                                prev_tag = all_tags[prev_tag_index]
                                                # transition probability of tags not found, do smoothing
                                                if (prev_tag, tag) not in trans_dic:
                                                    transition_prob = delta_transition / (trans_dic["flattened_matrix"].count(prev_tag) +
                                                                                          delta_transition * (trans_dic["unique_tag_count"] + 1))
                                                else:
                                                    transition_prob = trans_dic[(
                                                        prev_tag, tag)]
                                                # emission probability of tag not found for this given word, do smoothing
                                                if curr_word not in dic[tag]:
                                                    emission_prob = delta / \
                                                        (countyj + delta *
                                                         (num_words + 1))
                                                else:
                                                    emission_prob = dic[tag][curr_word]
                                                # if this is the last word, need to multiply by termination probability
                                                if i == len(test_words) - 1:
                                                    # termination probability of tag not found, do smoothing
                                                    if ('termination', str(tag)) not in trans_dic:
                                                        termination_prob = delta_transition / \
                                                            (trans_dic["total_tweets_count"] + delta_transition * (
                                                                trans_dic["unique_tag_count"] + 1))
                                                    else:
                                                        termination_prob = trans_dic[(
                                                            'termination', str(tag))]
                                                    curr_prob = prob * transition_prob * emission_prob * termination_prob
                                                else:
                                                    curr_prob = prob * transition_prob * emission_prob
                                                compare_prob.append(curr_prob)
                                            # finding the max probability to get to this current state and emitting this word
                                            pi_i_tag = max(compare_prob)
                                            index = compare_prob.index(
                                                pi_i_tag)
                                            # locating the previous tag that resulted in the highest probability
                                            backpointer = all_tags[index]
                                            # building this layer of transition matrix
                                            pi_i_tags.append(pi_i_tag)
                                            # building this layer of backpointer matrix
                                            backpointers.append(backpointer)

                                        # building the transition matrix
                                        trans_matrix.append(pi_i_tags)
                                        # building the backpointer matrix
                                        bp_matrix.append(backpointers)

                                    output_tags = []  # once transition and backpointer matrix have been built, generate outputs
                                    last_list = trans_matrix.pop()  # getting the last layer of transition matrix
                                    last_bp_list = bp_matrix.pop()  # getting the last layer of backpointer rmatrix
                                    # finding the maximum probability
                                    last_max_prob = max(last_list)
                                    max_index = last_list.index(last_max_prob)
                                    # finding the best tag with highest probability
                                    last_best_tag = all_tags[max_index]
                                    output_tags.append(last_best_tag)
                                    # locating the previous tag from the last layer of the backpointer matrix
                                    next_tag = last_bp_list[max_index]
                                    # storing the outputs
                                    output_tags.append(next_tag)

                                    while bp_matrix:  # find the backpointers until the backpointer matrix is empty
                                        # finding the index of the tag that was backpointed to
                                        next_index = all_tags.index(next_tag)
                                        prev_bp_list = bp_matrix.pop()  # getting the last layer of backpointer matrix
                                        # using the index to find the next backpointed tag
                                        next_tag = prev_bp_list[next_index]
                                        # adding the tag to list of outputs
                                        output_tags.append(next_tag)

                                    # since tags were backpointed, reverse the list to get the correct direction
                                    output_tags.reverse()

                                    # CHANGING TAGS FOR WORDS based on linguistical property
                                    indexesfor_at = []
                                    indexesfor_hashtag = []
                                    indexesfor_url = []
                                    indexes_for_dollar = []
                                    indexes_for_and = []
                                    indexes_for_RT = []
                                    indexes_for_D = []
                                    indexes_for_P = []
                                    punctuations = [
                                        '?', '!', '(', ')', '[', ']', '{', ',', '.', '"', '*']
                                    indexes_for_punctuations = []
                                    objects = ["i", "you", "me",
                                               "her", "him", "they", "them"]
                                    indexes_for_objects = []

                                    for i in range(len(test_words)):
                                        current_word = test_words[i]
                                        if current_word.startswith("@"):
                                            indexesfor_at.append(i)
                                        if current_word.startswith("#"):
                                            indexesfor_hashtag.append(i)
                                        if current_word.startswith("http:"):
                                            indexesfor_url.append(i)
                                        if current_word[-1].isdigit() or current_word[-1] == "%":
                                            indexes_for_dollar.append(i)
                                        if current_word == "and" or current_word == "&" or current_word == "but":
                                            indexes_for_and.append(i)
                                        if current_word == "RT":
                                            indexes_for_RT.append(i)
                                        if current_word == "a" or current_word == "the":
                                            indexes_for_D.append(i)
                                        if current_word == "for":
                                            indexes_for_P.append(i)
                                        for object in objects:
                                            if current_word == object:
                                                indexes_for_objects.append(i)
                                        for punctuation in punctuations:
                                            if punctuation in current_word:
                                                indexes_for_punctuations.append(
                                                    i)

                                    for index in indexes_for_punctuations:
                                        output_tags[index] = ","
                                    for index in indexes_for_and:
                                        output_tags[index] = "&"
                                    for index in indexes_for_RT:
                                        output_tags[index] = "~"
                                    for index in indexes_for_D:
                                        output_tags[index] = "D"
                                    for index in indexes_for_P:
                                        output_tags[index] = "P"
                                    for index in indexes_for_objects:
                                        output_tags[index] = "O"
                                    for index in indexes_for_dollar:
                                        output_tags[index] = "$"
                                    for index in indexesfor_url:
                                        output_tags[index] = "U"
                                    for index in indexesfor_hashtag:
                                        output_tags[index] = "#"
                                    for index in indexesfor_at:
                                        output_tags[index] = "@"

                                    while output_tags:  # writing the output to the output file
                                        output.write(output_tags.pop(0) + "\n")
                                output.write("\n")
                                trans_matrix = []  # this marks the end of the viterbi algorithm, clear the transition matrix, backpointer matrix and list of test words for the next viterbi iteration
                                bp_matrix = []
                                test_words = []
                                continue  # go on to the next tweet

                            else:  # there is no empty line encountered, add the current word to the list of test words
                                test_words.append(word.strip().lower())


viterbi_predict2("twitter_tags.txt", "trans_probs.txt", "output_probs.txt", "twitter_dev_no_tag.txt",
                 "viterbi_predictions2.txt", "twitter_train.txt")


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip()
                          for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip()
                             for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth:
            correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)


# print(evaluate("naive_predictions.txt", "twitter_dev_ans.txt"))
# print(evaluate("naive_predictions2.txt", "twitter_dev_ans.txt"))
print(evaluate("viterbi_predictions.txt", "twitter_dev_ans.txt"))
print(evaluate("viterbi_predictions2.txt", "twitter_dev_ans.txt"))

# def run():
#     '''
#     You should not have to change the code in this method. We will use it to execute and evaluate your code.
#     You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
#     uncomment them later.
#     This sequence of code corresponds to the sequence of questions in your project handout.
#     '''

#     ddir = ''  # your working dir

#     in_train_filename = f'{ddir}/twitter_train.txt'

#     naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

#     in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
#     in_ans_filename = f'{ddir}/twitter_dev_ans.txt'
#     naive_prediction_filename = f'{ddir}/naive_predictions.txt'
#     naive_predict(naive_output_probs_filename,
#                   in_test_filename, naive_prediction_filename)
#     correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
#     print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

#     naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
#     naive_predict2(naive_output_probs_filename, in_train_filename,
#                    in_test_filename, naive_prediction_filename2)
#     correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
#     print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

#     trans_probs_filename = f'{ddir}/trans_probs.txt'
#     output_probs_filename = f'{ddir}/output_probs.txt'

#     in_tags_filename = f'{ddir}/twitter_tags.txt'
#     viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
#     viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
#                     viterbi_predictions_filename)
#     correct, total, acc = evaluate(
#         viterbi_predictions_filename, in_ans_filename)
#     print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

#     trans_probs_filename2 = f'{ddir}/trans_probs2.txt'
#     output_probs_filename2 = f'{ddir}/output_probs2.txt'

#     viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
#     viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
#                      viterbi_predictions_filename2)
#     correct, total, acc = evaluate(
#         viterbi_predictions_filename2, in_ans_filename)
#     print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')


# if __name__ == '__main__':
#     run()
