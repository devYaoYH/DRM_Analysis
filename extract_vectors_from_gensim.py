import sys
import json
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

DATA_PATH = "cmr_word2vec.bin"
model = KeyedVectors.load(DATA_PATH)

word_dic = dict()
with open("lsa_tfr.txt", "r") as fin:
    for line in fin:
        # Grab corresponding vector from model
        w = line.strip()
        if (' ' in w):
            w = w.replace(' ', '_')
        if (w not in model.vocab.keys()):
            if (w.capitalize() not in model.vocab.keys()):
                print(f"Word {w} NOT FOUND", file=sys.stderr)
            else:
                word_dic[w] = [float(f) for f in model[w.capitalize()]]
        else:
            word_dic[w] = [float(f) for f in model[w]]

with open("cmr_murd62_w2v.json", "w+") as fout:
    json.dump(word_dic, fout)