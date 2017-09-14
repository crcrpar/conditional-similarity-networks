import glob
import os
import json

from tqdm import tqdm


def parse_tripletlist(path):
    indices_list = list()
    with open(path) as f:
        for line in f:
            a, b, c = list(map(int, line.strip().split(' ')))
            indices_list.append(a)
            indices_list.append(b)
            indices_list.append(c)

    return indices_list


def parse_split(root, split='train'):
    query = '*{}.txt'.format(split)
    query = os.path.join(root, query)
    split_tripletlist_files = glob.glob(query)
    indices_list = list()
    for path in split_tripletlist_files:
        tmp_indices_list = parse_tripletlist(path)
        indices_list.extend(tmp_indices_list)
    unique_indices_list = sorted(list(set(indices_list)))
    return unique_indices_list


def save_unique_split_indices(root, split, out_path, returns=False):
    unique_indices_list = parse_split(root, split)
    with open(out_path, 'w') as f:
        for line in unique_indices_list:
            f.write('{}\n'.format(line))
    if returns:
        return unique_indices_list


def lookup_json(indices_list, filenames_path):
    with open(filenames_path) as f:
        filenames = [line.rstrip('\n') for line in f]
    split_filenames = [filenames[index] for index in indices_list]
    return split_filenames


def save_unique_split_filenames(split_filenames, out_path):
    with open(out_path, 'w') as f:
        for filename in split_filenames:
            f.write('{}\n'.format(filename))


def save_unseen_filenames(root, out_path=None, returns=False):

    def load_json(path):
        with open(path) as f:
            return [line.strip() for line in f]

    train = set(load_json(os.path.join(root, 'train_filenames.json')))
    val_test = load_json(os.path.join(root, 'val_filenames.json')) + \
        load_json(os.path.join(root, 'test_filenames.json'))
    disjoint = set(val_test) - set(train)

    if out_path is None:
        out_path = os.path.join(root, 'not_training_filenames.json')
    with open(out_path, 'w') as f:
        for name in disjoint:
            f.write('{}\n'.format(name))
    if returns:
        return disjoint


def main():
    _root = 'data'
    base_path = 'tripletlists'
    root = os.path.join(_root, base_path)
    filenames_path = os.path.join(_root, 'filenames.json')
    out_path = 'splited_paths'
    out_root = os.path.join(_root, out_path)
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    split_list = ['train', 'val', 'test']
    for split in tqdm(split_list):
        out_path = os.path.join(_root, '{}_indices.json'.format(split))
        split_indices_list = save_unique_split_indices(
            root, split, out_path, True)
        split_filenames = lookup_json(split_indices_list, filenames_path)
        out_path = out_path.replace('indices', 'filenames')
        save_unique_split_filenames(split_filenames, out_path)
    save_unseen_filenames(_root, None, False)


if __name__ == '__main__':
    main()
