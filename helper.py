import numpy as np
from itertools import groupby

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


def get_tags(path):
    '''
        Get unique tags in dataset
    '''
    tags = set()

    with open(path) as fp:
        for empty, line in groupby(fp, lambda x: x.startswith('\n')):
            if not empty:
                labels = set(d.split()[1] for d in line if len(d.strip()))
                tags = tags.union(labels)
            
    return list(tags)


def create_str(prob_type, x, y):
  '''
    Create feature strings
  '''
  return f"{prob_type}:{x}+{y}"