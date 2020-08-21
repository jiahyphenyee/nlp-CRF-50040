import os
import numpy as np
from helper import create_str, load_data, parse_x_y

def estimate_emission_parameter(file_path):
         
    emissions = {}
    y_count = {}
    f = {}
    
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
            emissions[(x, y)]  = emissions.get((x,y),0) + 1
            emissions[(pos, y)] = emissions.get((pos,y),0) + 1

    for x, y in emissions.keys():
        key = "emission:" + y + "+" + x
        f[key] = np.log(emissions[(x, y)] / y_count[y])
        emissions[(x, y)] = np.log(emissions[(x, y)] / y_count[y])
    
    return f, list(all_y), emissions

train_path = full_path + "train"
f ,tags, emissions = estimate_emission_parameter(train_path)


def get_resulting_and_transition_dict(file_path, emission_dict):
    transitions = {}
    y_count = {}
    
    train_data = open(file_path, 'r', encoding="utf-8")
    lines = train_data.readlines()
    start = "start"
    all_y = set(["start", "stop"])

    for line in lines:
        line = line.strip()
        if len(line) <= 0:
            transitions[(start, "stop")] = transitions.get((start, "stop"),0) + 1
            start = "start"
            y_count[start] = y_count.get(start,0) + 1
        else:
            x, pos, y = line.split(" ")
            transitions[(start, y)] =  transitions.get((start,y),0) + 1
            y_count[y] = y_count.get(y,0) + 1
            start = y
            all_y.add(y)

    for start, end in transitions.keys():
        key = "transition:" + start + "+" + end
        emission_dict[key] = np.log(transitions[(start, end)] / y_count[start])
        transitions[(start, end)] = np.log(transitions[(start, end)] / y_count[start])

    return emission_dict, transitions

f, transitions = get_resulting_and_transition_dict(train_path, f)



def get_combined_dict(f, emissions, transitions):
    for word, state in emissions.keys():
        for start, end in transitions.keys():
            key = "combine:" + start + "+" + end + "+" + word
            f[key] = transitions.get((start, end), -9999999) + emissions.get((word, end), -9999999)
    return f

f = get_combined_dict(f, emissions, transitions)




def viterbi_algo(x, tags, f):
    scores = np.full((len(x), len(tags)), -np.inf)
    parents = np.full((len(x), len(tags)), 0, dtype=int)
    threshold = -9999999
    for i in range(len(tags)):
        combined_key1 = "combine:" + "start" + "+" + tags[i] + "+" + x[0].split()[0]
        combined_key2 = "combine:" + "start" + "+" + tags[i] + "+" + x[0].split()[1]
        emission_key1 = "emission:" + tags[i] + "+" + x[0].split()[0]
        emission_key2 = "emission:" + tags[i] + "+" + x[0].split()[1]
        transition_key = "transition:" + "start" + "+" + tags[i]

        scores[0, i] =+ f.get(combined_key1, threshold) + \
                        f.get(combined_key2, threshold) + \
                        f.get(emission_key1, threshold) + \
                        f.get(emission_key2,threshold) + \
                            f.get(transition_key, threshold) 

    for i in range(1, len(x)):
        for j in range(len(tags)): 
            for k in range(len(tags)):
                combined_key1 = "combine:" + tags[j] + "+" + tags[k] + "+" + x[i].split()[0]
                combined_key2 = "combine:" + tags[j] + "+" + tags[k] + "+" + x[i].split()[1]
                emission_key1 = "emission:" + tags[k] + "+" + x[i].split()[0]
                emission_key2 = "emission:" + tags[k] + "+" + x[i].split()[1]
                transition_key = "transition:" + tags[j] + "+" + tags[k]

                overall_score = scores[i-1, j] + f.get(combined_key1, threshold) + \
                                                    f.get(combined_key2, threshold) + \
                                                    f.get(emission_key1,threshold) + \
                                                    f.get(emission_key2, threshold) + \
                                                    f.get(transition_key, threshold)
                   
                if overall_score > scores[i, k]:
                    scores[i, k] = overall_score
                    parents[i,k] = j
    
    best_score = -np.inf
    best_parent = None


    for i in range(len(tags)):
        transition_key = "transition:" + tags[i] + "+" + "stop"
        total = scores[len(x)-1, i] + f.get(transition_key, -10**8)    
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


def get_evaluation(data_path,f):

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
        
    out_path = os.path.join(data_path, "dev.p5.CRF.f4.out")
    out_file = open(out_path, "w", encoding="utf-8")
    
    for x in sequences:
        predicted = viterbi_algo(x, tags, f)
        for i in range(len(x)):
            out_file.write(x[i] + ' ' + predicted[i] + '\n')
        out_file.write('\n')

    out_file.close()
    


if __name__ == '__main__':
    full_path = '/data/full/'
    full_train_data = load_data(full_path + 'train')
    full_dev_out_data = load_data(full_path + 'dev.out')

    f , tags, emissions = estimate_emission_parameter(full_path + "train")
    f, transitions = get_resulting_and_transition_dict(full_path + "train", f)
    f = get_combined_dict(f, emissions, transitions)

    get_evaluation(full_path, f)