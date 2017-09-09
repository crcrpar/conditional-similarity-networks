import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns


def parse_log(path):
    with open(path) as f:
        for line in f:
            line = line.strip(' ')
            if line.startswith('Test'):
                # TODO
            elif line.startswith('Train'):
                # TODO
            else:
                continue
