import numpy as np
from itertools import groupby
from conlleval import evaluate

def load_data(path):
    '''
        Load dataset from input path into numpy array
    '''
    out = []

    with open(path) as fp:
        for empty, line in groupby(fp, lambda x: x.startswith('\n')):
            if not empty:
                out.append(np.array([[str(x) for x in d.split()] for d in line if len(d.strip())]))
  
    return out


def parse_x_y(data):
  '''
    Separate dataset into list of sentences and list of corresponding tag sequences
  '''
  x = []
  y = []

  for i in range(len(data)):
    x.append(data[i][:, 0].tolist())
    y.append(data[i][:, 1].tolist())

  return x, y


def parse_x_y_z(data):
    '''
        Separate dataset into list of sentences and list of corresponding tag sequences and pos
    '''
    x = []
    y = []
    z = []

    for i in range(len(data)):
        x.append(data[i][:, 0].tolist())
        y.append(data[i][:, 1].tolist())
        z.append(data[i][:, 2].tolist())

    return x, y, z


def get_tags(path, ind):
    '''
        Get unique tags in dataset
    '''
    tags = set()

    with open(path) as fp:
        for empty, line in groupby(fp, lambda x: x.startswith('\n')):
            if not empty:
                labels = set(d.split()[ind] for d in line if len(d.strip()))
                tags = tags.union(labels)

    return list(tags)


def create_str(prob_type, x, y):
    '''
        Create feature strings
    '''
    return f"{prob_type}:{x}+{y}"


def get_evaluate(true_data, pred_tags):
    _, ty = parse_x_y(true_data)

    true_tags = []

    for tag_seq in ty:
        true_tags = true_tags + tag_seq

    evaluate(true_tags, pred_tags)


def get_evaluate_pos(true_data, pred_tags):
    _, _, tz = parse_x_y_z(true_data)

    true_tags = []

    for tag_seq in tz:
        true_tags = true_tags + tag_seq

    evaluate(true_tags, pred_tags)