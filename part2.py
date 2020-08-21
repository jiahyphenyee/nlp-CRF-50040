import numpy as np
from collections import defaultdict

from helper import create_str, load_data

def get_crf_score(x, y, f_weights):
    '''
        Compute CRF scores given pair of input and output sequence pair (x, y)
        param:
            x: list[str] --- complete input word sentence
            y: list[str] --- complete output label sequence
            f_weights: dict{str(x,y): int} --- a dict mapping feature to weight 
        return:
            score: float
    '''
    assert(len(x) == len(y))

    f_count = defaultdict(int)
    score = 0

    # emission
    for i in range(len(x)):
        emission_s = create_str("emission", y[i], x[i])
        f_count[emission_s] += 1

    # transition
    y_start_stop = y + ["STOP"]
    y_prev = "START"
    for yy in y_start_stop:
        transition_s = create_str("transition", y_prev, yy)
        f_count[transition_s] += 1
        y_prev = yy
        
    # final score - sum of all weights of involved features
    for key, count in f_count.items():
        weight = f_weights[key]
        score += weight * count

    return score


def viterbi(x, tags, f):
    '''
        Viterbi decoder for input sequence x
        param:
            x: list[str] --- complete input word sentence
            tags: list[str] --- list of all unique tags (y) from dataset
            f: dict{str(x,y): int} --- a dict mapping feature to weight 
        return:
            y: list[str] --- most probable output tag sequence
    '''
    n = len(x)
    scores = np.full((n, len(tags)), -np.inf) # store all scores
    backpointers = np.full((n, len(tags)), tags.index('O'), dtype=np.int) # default backpointer to 'O'
    best_score = None
    best_backpointer = None

    # first node
    for ind, tag in enumerate(tags):
        transition_s = create_str("transition", "START", tag)
        emission_s = create_str("emission", tag, x[0])
        scores[0, ind] = f.get(transition_s, -np.inf) + f.get(emission_s, -np.inf)

    # rest of the nodes
    for i in range(1, n):
        for j, tag_j in enumerate(tags):
            for k, tag_k in enumerate(tags):
                transition_s = create_str("transition", tag_j, tag_k)
                emission_s = create_str("emission", tag_k, x[i])
                current_score = f.get(transition_s, -np.inf) + f.get(emission_s, -np.inf) + scores[i-1, j]
                # print("prev scores:", scores[i,k])
                # print("current score:", current_score)

                # update
                if current_score > scores[i,k]:
                    scores[i,k] = current_score
                    backpointers[i,k] = j
    
    # STOP node
    for ind, tag in enumerate(tags):
        transition_s = create_str("transition", tag, "STOP")
        current_score = scores[n-1, ind] + f.get(transition_s, -np.inf)

        if best_score == None or current_score > best_score:
            best_score = current_score
            best_backpointer = ind
    
    # backtrack
    current_bp = best_backpointer
    y = [tags[best_backpointer]]
    
    for i in range(n-1, 0, -1):
        current_bp = backpointers[i, current_bp]
        y.insert(0, tags[current_bp])

    return y