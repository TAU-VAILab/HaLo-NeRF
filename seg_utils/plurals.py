import numpy as np

def to_plur(label):
    if label[-1] == 'y' and label[-2] not in 'aeiou':
        return label[:-1] + 'ies'
    if label[-1] in 'sz' or label.endswith('ch') or label.endswith('sh'):
        return label + 'es'
    return label + 's'

def to_plur_aug(label):
    if np.random.random() > 0.5:
        return to_plur(label)
    return label