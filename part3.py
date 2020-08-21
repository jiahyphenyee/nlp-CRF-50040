import numpy as np
from collections import defaultdict

from helper import create_str, load_data, parse_x_y
from part2 import get_crf_score


def get_forward(x, tags, f):
    '''
        Compute 2nd term in loss function for an input sequence, using the forward algorithm
        param:
            x: list[str] --- input sentence
            tags: list[str] --- list of all unique tags (y) from dataset
            f: dict{str(x,y): int} --- a dict mapping feature to weight
        return:
            scores: np.array --- forward scores
            alpha: float --- forward score for input sequence
    '''
    n = len(x)
    scores = np.zeros((n, len(tags)))
    sum_scores = 0

    # first node
    for ind, tag in enumerate(tags):
        transition_s = create_str("transition", "START", tag)
        emission_s = create_str("emission", tag, x[0])
        scores[0, ind] = f.get(transition_s, -10**8) + f.get(emission_s, -10**8)

    # rest of the  nodes
    for i in range(1, n):
        for j, tag_j in enumerate(tags):
            current_score = 0

        for k, tag_k in enumerate(tags):
            transition_s = create_str("transition", tag_k, tag_j)
            emission_s = create_str("emission", tag_j, x[i])
            # summation
            current_score += np.exp(min(f.get(transition_s, -10**8) + f.get(emission_s, -10**8) + scores[i-1, k], 700))
            
        # update score
        scores[i, j] = np.log(current_score) if current_score else -10**8

    # STOP node
    for ind, tag in enumerate(tags):
        transition_s = create_str("transition", tag, "STOP")
        current_score = np.exp(scores[n-1, ind] + f.get(transition_s, -10**8))

        # sum to get forward prob
        sum_scores += current_score
    
    alpha = np.log(sum_scores)

    return scores, alpha


def compute_crf_loss(data, tags, f, nabla=None):
    '''
        Compute CRF loss of a training set
        param:
            data: list[list[list[word, tag]]] --- data set as list of words with respective tags
            tags: list[str] --- list of all unique tags (y) from dataset
            f: dict{str(x,y): int} --- a dict mapping feature to weight
        return:
            float --- forward score for input sequence
    '''
    loss = 0
    x, y = parse_x_y(data)

    assert len(x) == len(y)

    for i in range(len(x)):
        crf_score = get_crf_score(x[i], y[i], f)
        _, forward_score = get_forward(x[i], tags, f)
        l = -(crf_score - forward_score)
        loss += l

    # L2 Regularization
    if nabla is not None:
        reg_loss = 0
        for feature_key in f:
            reg_loss += f[feature_key]**2
        reg_loss = nabla*reg_loss
        loss += reg_loss

    return loss


def get_backward(x, tags, f):
    '''
        Compute backward scores
        param:
            x: list[str] --- input sentence
            tags: list[str] --- list of all unique tags (y) from dataset
            f: dict{str(x,y): int} --- a dict mapping feature to weight
        return:
            scores: np.array --- backwards scores
            beta: float --- backward score for input sequence
    '''
    n = len(x)
    scores = np.zeros((n, len(tags)))
    sum_score = 0
    
    # STOP transition
    for ind, tag in enumerate(tags):
        transition_s = create_str("transition", tag, "STOP")
        scores[n-1, ind]  = f.get(transition_s, -10**8)

    for i in range(n-1, 0, -1):
        for k, tag_k in enumerate(tags):
            current_score = 0

            for j, tag_j in enumerate(tags):
                transition_s = create_str("transition", tag_k, tag_j)
                emission_s = create_str("emission", tag_j, x[i])

            current_score += np.exp(min(f.get(emission_s, -10**8) + f.get(transition_s, -10**8) + scores[i, j], 700))

            # update scores array
            scores[i-1, k] = np.log(current_score) if current_score else -10**8
            

    # START
    for ind, tag in enumerate(tags):
        transition_s = create_str("transition", "START", tag)
        emission_s = create_str("emission", tag, x[0])
        
        current_score = np.exp(min(f.get(emission_s, -10**8) + f.get(transition_s, -10**8) + scores[0, ind], 700))
        sum_score += current_score

    beta = np.log(sum_score) if sum_score else -700
    
    return scores, beta


def get_expected_count(x, tags, f):
    '''
        Forward-backward algorithm to compute the expected counts for an input sequence
        param:
            x: list[str] --- input sentence
            tags: list[str] --- list of all unique tags (y) from dataset
            f: dict{str(x,y): int} --- a dict mapping feature to weight
        return:
            f_e_counts: dict --- expected count for each feature.
    '''

    n = len(x)
    f_e_counts = defaultdict(float)
    
    forward_scores, alpha = get_forward(x, tags, f)
    # forward_prob = np.exp(min(alpha, 700))
    backward_scores, beta = get_backward(x, tags, f)
    # backward_prob = np.exp(min(beta, 700))

    # START & STOP expected transition counts
    for j, tag in enumerate(tags):
        start_s = create_str("transition", "START", tag)
        stop_s = create_str("transition", tag, "STOP")
        f_e_counts[start_s] += np.exp(min(forward_scores[0, j] + backward_scores[0, j] - alpha, 700))
        f_e_counts[stop_s] += np.exp(min(forward_scores[n-1, j] + backward_scores[n-1, j] - alpha, 700))

    # expected emission counts
    for i in range(n):
        for j, tag_j in enumerate(tags):
            emission_s = create_str("emission", tag_j, x[i])
            f_e_counts[emission_s] += np.exp(min(forward_scores[i, j] + backward_scores[i, j] - alpha, 700))
    
    # expected transition probabilities 
    for j, tag_j in enumerate(tags):
        for k, tag_k in enumerate(tags):
            prob = 0
            transition_s = create_str("transition", tag_j, tag_k)
        
        for i in range(n-1):
            emission_s = create_str("emission", tag_k, x[i+1])

            prob += np.exp(min(forward_scores[i, j] + backward_scores[i+1, k] + f.get(transition_s, -10**8) + f.get(emission_s, -10**8) - alpha, 700))

        f_e_counts[transition_s] = prob

    return f_e_counts


def get_actual_count(x, y, f):
    ''' 
    Inputs:
        x: list[str] --- input sentence
        y: list[str] --- tag sequence
        f: dict{str(x,y): int} --- a dict mapping feature to weight
    Outputs:
        f_a_counts: dict --- actual count for each feature
    '''
    
    assert(len(x) == len(y))
    n = len(x) 
    
    f_a_counts = defaultdict(int)
    
    # emission features
    for i in range(n):
        emission_s = create_str("emission", y[i], x[i])
        f_a_counts[emission_s] += 1
    
    # transition features
    start_stop_y = ["START"] + y + ["STOP"]
    for i in range(1, n+2):
        transition_s = create_str("transition", start_stop_y[i-1], start_stop_y[i])
        f_a_counts[transition_s] += 1
    
    return f_a_counts


def compute_gradients(data, tags, f, nabla=None):
    '''
        Get gradients of each feature and store in a dictionary
        param:
            data: list[list[list[word, tag]]] --- data set as list of words with respective tags
            tags: list[str] --- list of all unique tags (y) from dataset
            f: dict{str(x,y): int} --- a dict mapping feature to weight
        return:
            f_gradients: dict{str(x,y): float} --- dict mapping feature to gradient (forward-backward)
    '''
    
    list_x, list_y = parse_x_y(data)
    f_gradients = defaultdict(float)

    assert len(list_x) == len(list_y)

    # for each input sentence and corresponding tag sequence
    for i in range(len(list_y)):
        x = list_x[i]
        y = list_y[i]

        f_e_counts = get_expected_count(x, tags, f)
        f_a_counts = get_actual_count(x, y, f)

        for feature, ec in f_e_counts.items():
            f_gradients[feature] += ec
        
        for feature, ac in f_a_counts.items():
            f_gradients[feature] -= ac

    # L2 regularization
    if nabla is not None:
        for k, _ in f.items():
            f_gradients[k] += 2*nabla*f[k]

    return f_gradients

