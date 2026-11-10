import numpy as np
from scipy.stats import rankdata
def col_auc(Atr, y):
    """AUROC of every column for y=1 against y=0 (rank formula, ties get mid-ranks)."""
    r = rankdata(Atr, axis=0); n1 = y.sum(); return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
