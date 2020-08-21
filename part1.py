import numpy as np
from collections import defaultdict

from helper import create_str, load_data


def get_emission_scores(data):
    '''
        Compute emission scores
        param:
            data: list[list[list[word, tag]]] --- list of sentences as list of words with respective tags
        return:
            f: dict{str(x,y): int} --- a dict mapping string to emission score 
    '''

    f = {} 
    emission_count = defaultdict(int)
    tag_count = defaultdict(int) 

    for sent in data:
        for x, y in sent:
            emission_count[(x, y)] += 1
            tag_count[y] += 1

    for x, y in emission_count:
        s = create_str("emission", y, x)
        e_prob = emission_count[(x, y)] / tag_count[y]

        f[s] = np.log(e_prob)
    
    return f


def get_transition_scores(data):
  '''
    Compute transition scores
    param:
        data: list[list[list[word, tag]]] --- list of sentences as list of words with respective tags
    return:
        f: dict{str(x,y): int} --- a dict mapping string to transition score 
  '''

  f = {}
  transition_count = defaultdict(int)
  tag_count = defaultdict(int) 

  for sent in data:
    y_prev = "START"
    tag_count[y_prev] += 1

    for _, y in sent:
      transition_count[(y_prev, y)] += 1
      tag_count[y] += 1
      y_prev = y

    # update end of sentence
    transition_count[(y_prev, "STOP")] += 1

  for y_prev, y in transition_count:
    s = create_str("transition", y_prev, y)
    t_prob = transition_count[(y_prev, y)] / tag_count[y_prev]

    f[s] = np.log(t_prob)

  return f


def get_emis_trans_dict(data):
    '''
        Combine emission and transition dictionaries
    '''
    f_emission = get_emission_scores(data)
    f_transition = get_transition_scores(data)

    f = {**f_emission, **f_transition}
    assert(len(f) == len(f_emission) + len(f_transition))

    return f

if __name__ == '__main__':
    partial_path = '/data/partial/'
    partial_train_data = load_data(partial_path + 'train')

    f = get_emis_trans_dict(partial_train_data)