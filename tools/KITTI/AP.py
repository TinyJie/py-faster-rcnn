#!/usr/bin/env python

import sys
import os.path as osp

class_name = ['car', 'pedestrian', 'cyclist']
result_dir = sys.argv[1]
recall = ['0.000000', '0.100000', '0.200000', '0.300000', '0.400000', '0.500000', '0.600000', '0.700000', '0.800000', '0.900000', '1.000000']

for cls in class_name:
    filename = osp.join(result_dir, cls + '_detection.txt')
    easy = []
    modest = []
    hard = []
    with open(filename) as f:
        for line in f:
            words = line.split()
            if words[0] in recall:
                easy.append(float(words[1]))
                modest.append(float(words[2]))
                hard.append(float(words[3]))
        print '{} [easy] AP = {} [modest] AP ={} [hard] AP = {}'.format(cls, sum(easy)/len(easy), sum(modest)/len(modest), sum(hard)/len(hard))


