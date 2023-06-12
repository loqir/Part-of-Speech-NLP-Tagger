delta_transition = 0.1

with open("twitter_train.txt", "r", encoding="utf8") as input:
    word_matrix = []
    tag_matrix = []
    tag_set = set()
    dic = {}
    word_list = []
    tag_list = []
    count = 0
    number_of_HMM = 0

    for line in input:
        if line != "\n":
            count += 1
            both = line.split()
            if len(both) > 0:
                word = both[0]
                tag = both[1]
                word_list.append(word)
                tag_list.append(tag)
                tag_set.add(tag)
        else:
            word_matrix.append(word_list)
            tag_matrix.append(tag_list)
            word_list = []
            tag_list = []

    for HMM in tag_matrix:
        number_of_HMM += 1
        starting_tag = HMM[0]
        if ("initial", starting_tag) not in dic:
            dic[("initial", starting_tag)] = 1
        else:
            dic[("initial", starting_tag)] += 1

    for HMM in tag_matrix:
        ending_tag = HMM[-1]
        if ("termination", ending_tag) not in dic:
            dic[("termination", ending_tag)] = 1
        else:
            dic[("termination", ending_tag)] += 1

    dic["total_tweets_count"] = number_of_HMM
    dic["unique_tag_count"] = len(tag_set)

    flattened_matrix = []
    for tag_list in tag_matrix:
        for tag in tag_list:
            flattened_matrix.append(tag)

    for key in dic:  # smoothing
        dic[key] += delta_transition
        if key[0] == "initial":
            dic[key] /= (number_of_HMM + delta_transition * (len(tag_set) + 1))
        else:
            dic[key] /= (flattened_matrix.count(key[1]) +
                         delta_transition * (len(tag_set) + 1))

    for tag_list in tag_matrix:
        for i in range(len(tag_list) - 1):
            current_tag = tag_list[i]
            next_tag = tag_list[i + 1]
            if (current_tag, next_tag) not in dic:
                dic[(current_tag, next_tag)] = 1
            else:
                dic[(current_tag, next_tag)] += 1

    for key in dic:
        if len(key) == 2:
            key_1 = key[0]
            key_2 = key[1]
            if len(key_1) == 1 & len(key_2) == 1:
                dic[key] += delta_transition  # smoothing
                dic[key] /= (flattened_matrix.count(key_1) +
                             delta_transition * (len(tag_set) + 1))

    dic["flattened_matrix"] = flattened_matrix


with open("trans_probs.txt", 'w', encoding="utf8") as output:
    output.write(str(dic))


delta = 0.1
smooth_dic = {}
words_set = set()
with open("twitter_train.txt", "r", encoding="utf8") as input:
    for line in input:
        both = line.split()
        if len(both) > 0:
            word = both[0]
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


with open("output_probs.txt", 'w', encoding="utf8") as outputSMOOTH:
    outputSMOOTH.write(str(smooth_dic))
