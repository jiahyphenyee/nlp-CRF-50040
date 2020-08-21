import numpy as np
import os
from collections import defaultdict

from helper import create_str, load_data

def get_emission_dict(file_path):
    
    e = {}
    y_count = {}
    emission_dict = {}
    
    train_data = open(file_path, 'r', encoding="utf-8")
    lines = train_data.readlines()
    
    all_x = set()
    all_y = set()

    for line in lines:
        line = line.strip()

        if len(line) > 0:
            x, pos, y = line.split(" ")
            all_x.add(x)
            all_y.add(y)
            y_count[y] = y_count.get(y,0) + 1
            e[(x, y)]  = e.get((x,y),0) + 1
            e[(pos, y)] = e.get((pos,y),0) + 1

    for x, y in e.keys():
        key = create_str("emission", y, x)
        emission_dict[key] = np.log(e[(x, y)] / y_count[y])
    
    return emission_dict, list(all_y)


def viterbi_algo(x, tags, f):
    scores = np.full((len(x), len(tags)), -np.inf)
    parents = np.full((len(x), len(tags)), 0, dtype=int)
    
    for i in range(len(tags)):
        emission_key1 = "emission:" + tags[i] + "+" + x[0].split()[0]
        emission_key2 = "emission:" + tags[i] + "+" + x[0].split()[1]
        transmission_key = "transition:" + "start" + "+" + tags[i]
        scores[0, i] = f.get(emission_key1, -10e8) + f.get(emission_key2, -10e8) + f.get(transmission_key, -10e8)
    
    for i in range(1, len(x)):
        for j in range(len(tags)):
            for k in range(len(tags)):
                emission_key1 = "emission:" + tags[k] + "+" + x[i].split()[0]
                emission_key2 = "emission:" + tags[k] + "+" + x[i].split()[1]
                transmission_key = "transition:" + tags[j] + "+" + tags[k]
                overall_score = scores[i-1, j] + f.get(emission_key1, -10e8) + f.get(emission_key2, -10e8) + f.get(transmission_key, -10e8)

                if overall_score > scores[i, k]:
                    scores[i, k] = overall_score
                    parents[i,k] = j
    
    best_score = -np.inf
    best_parent = None
    
    for i in range(len(tags)):
        t_feature = "transition:" + tags[i] + "+" + "stop"
        total = scores[len(x)-1, i] + f.get(t_feature, -10**8)
        
        if total > best_score:
            best_score = total
            best_parent = i
    
    best_state = [tags[best_parent]]
    prev_parent = best_parent
    for i in range(len(x)-1, 0, -1):
        prev_parent = parents[i, prev_parent]
        output = tags[prev_parent]
        best_state = [output] + best_state
    
    return best_state

def get_f(file_path, emission_dict):

    t = {}
    y_count = {}
    
    train_data = open(file_path, 'r', encoding="utf-8")
    lines = train_data.readlines()
    start = "start"
    all_y = set(["start", "stop"])

    for line in lines:
        line = line.strip()
        if len(line) <= 0:
            t[(start, "stop")] = t.get((start,"stop"),0) + 1
            start = "start"
            y_count[start] = y_count.get(start,0) + 1
        else:
            x, pos, y = line.split(" ")
            t[(start, y)] = t.get((start,y),0) + 1
            y_count[y] = y_count.get(y,0) + 1
            start = y
            all_y.add(y)

    for start, end in t.keys():
        key = "transition:" + start + "+" + end
        emission_dict[key] = math.log(t[(start, end)] / y_count[start])

    return emission_dict


def get_prediction(data_path, resulting_dict):
    train_path = os.path.join(data_path, "train")
    input_path = os.path.join(data_path, "dev.in")
    
    test_set = open(input_path, 'r', encoding="utf-8")
    lines = test_set.readlines()
    
    sequences = []
    sequence = []
    for line in lines:
        if line == '\n':
            sequences.append(sequence)
            sequence = []
            continue

        line = line.replace('\n', '')
        sequence.append(line)

    out_path = os.path.join(data_path, "dev.p5.SP.out")
    out_file = open(out_path, "w", encoding="utf-8")
    
    for x in sequences:
        predicted = viterbi_algo(x, tags, resulting_dict)
        for i in range(len(x)):
            out_file.write(x[i] + ' ' + predicted[i] + '\n')
        out_file.write('\n')
    
    out_file.close()
        

def structured_perceptron(file_dir,resulting_dict,epoch = 15, lr= 0.01):
    train_path = os.path.join(file_dir, "train")
    input_path = os.path.join(file_dir, "dev.in")
    test_set = open(train_path, 'r', encoding="utf-8")
    lines = test_set.readlines()
    copy_dict = resulting_dict.copy()
    for n in range(epoch):
        sequences = [] #ls of sequences
        word_sequence = []
        correct_state = []
        for line in lines:
            temp = []
            if line == '\n':
                sequences.append([word_sequence,correct_state])
                word_sequence = []
                correct_state = []
                continue
            line = line.strip().split(" ")
            word_sequence.append(line[0]+" "+line[1])
            correct_state.append(line[2])
        for x in sequences:
            predicted = viterbi_algo(x[0], tags, copy_dict)
            x.append(predicted)
        for word_no in range(len(sequences)):
            sentence = sequences[word_no][0]
            word_only = []
            pos_only = []
            for word_pos in sentence:
                word,pos = word_pos.split(" ")
                word_only.append(word)
                pos_only.append(pos)
            correct_states = sequences[word_no][1]
            predicted_states = sequences[word_no][2]
            #for each prediction, check if its correct
            for i in range(1,len(word_only)):
                if correct_states[i] != predicted_states[i]:
#                     print(correct_states[i],predicted_states[i])
                    copy_dict["emission:"+ predicted_states[i] +"+"+ word_only[i]] -= 1* lr
                    copy_dict["emission:"+ predicted_states[i] +"+"+ pos_only[i]] -= 1*lr
                    copy_dict["transition:"+ predicted_states[i-1] +"+" + predicted_states[i]] -= 1*lr
                    copy_dict["emission:"+ correct_states[i] +"+"+ word_only[i]] += 1*lr
                    copy_dict["emission:"+ correct_states[i] +"+"+ pos_only[i]] += 1*lr
                    copy_dict["transition:"+ correct_states[i-1] +"+"+ correct_states[i]] += 1*lr

    return copy_dict


if __name__ == '__main__':
    full_path = '/data/full/'
    full_train_data = load_data(full_path + 'train')
    full_dev_out_data = load_data(full_path + 'dev.out')

    emission_dict, tags  = get_emission_dict(full_path + "train")
    f = get_f(full_path + "train", emission_dict)
    get_prediction(full_path, f)