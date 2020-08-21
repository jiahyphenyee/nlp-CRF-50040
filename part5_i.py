from helper import load_data, create_str, get_tags, get_evaluate_pos
import numpy as np
from itertools import groupby
from collections import defaultdict

def get_emission_scores_pos(data):
    '''
        Compute emission scores with POS features
        param:
            data: list[list[list[word, pos, tag]]] --- list of sentences as list of words with respective pos and tags
        return:
            f: dict{str(x,y): int} --- a dict mapping string to emission score 
    '''
    f = {} 
    emission_count = defaultdict(int)
    tag_count = defaultdict(int) 
    test = 0

    for sent in data:
        for x, y, z in sent:
            emission_count[(x, z)] += 1
            emission_count[(y, z)] += 1   # for POS
            tag_count[z] += 1
            test += 1

    for x, y in emission_count:
        s = create_str("emission", y, x)
        e_prob = emission_count[(x, y)] / tag_count[y]

        f[s] = np.log(e_prob)
    
    return f


def get_transition_scores_pos(data):
    '''
        Compute transition scores with POS features
        param:
            data: list[list[list[word, pos, tag]]] --- list of sentences as list of words with respective pos and tags
        return:
            f: dict{str(x,y): int} --- a dict mapping string to transition score 
    '''

    f = {}
    transition_count = defaultdict(int)
    tag_count = defaultdict(int) 

    for sent in data:
        z_prev = "START"
        tag_count[z_prev] += 1

        for x, y, z in sent:
            transition_count[(z_prev, z)] += 1
            tag_count[z] += 1
            z_prev = z

        # update end of sentence
        transition_count[(z_prev, "STOP")] += 1

    for y_prev, y in transition_count:
        s = create_str("transition", y_prev, y)
        t_prob = transition_count[(y_prev, y)] / tag_count[y_prev]

        f[s] = np.log(t_prob)

    return f


def viterbi_decode_pos(x, states, feature_dict):
    '''
    Inputs:
        x (list[str]): Input sequence.
        states (list[str]): Possible output states.
        feature_dict (dict[str] -> float): Dictionary that maps a given feature to its score.
    Outputs:
        y (list[str]): Most probable output sequence.
    '''
    
    n = len(x) # Number of words
    d = len(states) # Number of states
    scores = np.full((n, d), -np.inf) # Default state is -inf for missing values
    bp = np.full((n, d), 0, dtype=np.int) # Set default backpointer to the default_index state

    # Convert to lowercase
    x = [x[i] for i in range(n)]
    
    # Compute START transition scores
    for i, current_y in enumerate(states):
        transition_key = create_str("transition", "START", current_y)
        emission_key1 = create_str("emission", current_y, x[0].split()[0])
        emission_key2 = create_str("emission", current_y, x[0].split()[1])
        transmission_score = feature_dict.get(transition_key, -10**8)
        emission_score1 = feature_dict.get(emission_key1, -10**8)
        emission_score2 = feature_dict.get(emission_key2, -10**8)
        scores[0, i] = transmission_score + emission_score1 + emission_score2
    
    # Recursively compute best scores based on transmission and emission scores at each node
    for i in range(1, n):
        for k, prev_y in enumerate(states):
            for j, current_y in enumerate(states):
                transition_key = create_str("transition", prev_y, current_y)
                emission_key1 = f"emission:{current_y}+{x[i].split()[0]}"
                emission_key2 = f"emission:{current_y}+{x[i].split()[1]}"
                
                transition_score = feature_dict.get(transition_key, -10**8)
                emission_score1 = feature_dict.get(emission_key1, -10**8)
                emission_score2 = feature_dict.get(emission_key2, -10**8)
                overall_score = emission_score1 + emission_score2 + transition_score + scores[i-1, k]

                # Better score is found: Update backpointer and score arrays
                if overall_score > scores[i, j]:
                    scores[i, j] = overall_score
                    bp[i,j] = k
    
    # Compute for STOP
    highest_score = -np.inf
    highest_bp = None
    
    for j, prev_y in enumerate(states):
        transition_key = f"transition:{prev_y}+STOP"
        transition_score = feature_dict.get(transition_key, -10**8)
        overall_score = transition_score + scores[n-1, j]
        
        if overall_score > highest_score:
            highest_score = overall_score
            highest_bp = j
    
    # Follow backpointers to get output sequence
    result = [states[highest_bp]]
    prev_bp = highest_bp
    for i in range(n-1, 0, -1):
        prev_bp = bp[i, prev_bp]
        output = states[prev_bp]
        # Prepend result to output list
        result = [output] + result
    
    return result


def viterbi_dev_pos(path, tags, f_weights, output_file):
    '''
        Write Viterbi predicted tag sequences onto file, given input sentences
        param:
            path: str --- path to input sentences
            tags: list[str] --- list of all unique tags (y) from dataset
            f_weights: dict{str(x,y): int} --- a dict mapping feature to weight 
            output_file: str --- output file name
        return:
            pred_tags: list[str] --- combined list of tag sequences
    '''
    sentences = []
    pred_tags = []

    # Write predictions to file
    output_filename = str(path).replace("dev.in", output_file)

    # Read from dataset path
    with open(path) as f:
        lines = f.readlines()
        sentence = []
        
        for line in lines:
            formatted_line = line.strip()
            
            # Not the end of sentence, add it to the list
            if len(formatted_line) > 0:
                sentence.append(formatted_line)
            else:
                # End of sentence
                sentences.append(sentence)
                sentence = []

    # Write output file
    with open(output_filename, "w") as out:
        for x in sentences:
            # Run predictions
            pred_sentence = viterbi_decode_pos(x, tags, f_weights)
            
            # Write original word and predicted tags
            for i in range(len(x)):
                pred_tags.append(pred_sentence[i])
                out.write(x[i] + " " + pred_sentence[i] + "\n")
            
            # End of sentence, write newline
            out.write("\n")

    return pred_tags


def get_emis_trans_dict_pos(data):
    '''
        Combine emission and transition dictionaries
    '''
    f_emission_pos = get_emission_scores_pos(full_train_data)
    f_transition_pos = get_transition_scores_pos(full_train_data)

    f_pos = {**f_emission_pos, **f_transition_pos}
    assert(len(f_pos) == len(f_emission_pos) + len(f_transition_pos))

    return f_pos


if __name__ == '__main__':
    full_path = '/data/full/'
    full_train_data = load_data(full_path + 'train')
    full_dev_out_data = load_data(full_path + 'dev.out')
    tags_full = get_tags(full_path + 'train', 2)

    f_pos = get_emis_trans_dict_pos(full_train_data)

    p5i_tags = viterbi_dev_pos(full_path + "dev.in", tags_full, f_pos, 'dev.p5.CRF.f3.out')
    get_evaluate_pos(full_dev_out_data, p5i_tags)