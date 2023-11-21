import numpy as np
from tqdm import tqdm

import lib

lines = []
line_lengths = []
with open(lib.RETRAINING_DATA_PATH / "wikipedia/train.txt.original", "r") as f:
    for l in tqdm(f.readlines()):
        l = l.strip()
        lines.append(l)
        spl = l.split(" ")
        line_lengths.append(len(spl))

lens = np.array(line_lengths)
print(lens.mean())
print(lens.max())
print(lens.std())
print(np.median(lens))
