"""SO-HN (main §2.3; owner K): the importable name of the H-neuron machinery for parts L and D (implemented in r1b_k_hn.py and
r1b_k_base.py): the hook manager (Tap: z and CETT on the down_proj input; Interv: scale / clamp / add on z), SetReader (a_X = mean
zCETT over a set, SO-CHH), wnorms, refw_stats, percentiles, cett_features, the GPU FISTA selector select(X, y, C, exclude_mask), c0,
rand_sets (same count per layer, mean answer-CETT within +-20 %), match_sets (damage matching), jaccard."""
from r1b_k_base import Tap, Interv, SetReader, wnorms, by_layer, set_values  # noqa: F401
from r1b_k_hn import (refw_stats, percentiles, cett_features, select, fista, c0, rand_sets, match_sets, jaccard,  # noqa: F401
                      neurons_of, cols_of, l2_detector)
