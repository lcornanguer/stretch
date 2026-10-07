import torch
import numpy as np
from ..edge_fitting import fit_edge
from itertools import permutations

class MemoizedEdgeScore:

    def __init__(self, X, global_params, verbosity=False):
        self.verbosity = verbosity
        self.X = X

        # Memoized info
        self.mdl_cache = {}
        self.model_cache = {}
        self.idl_cache = {}

        self.curr_graph = None

        self.global_params = global_params

    def score_edge(self, j, pa) -> int:
        """
        Evaluates score for a causal relationship pa(Xj)->Xj.

        :param j: Xj
        :param pa: pa(Xj)
        :return: score_up=score(Xpa->Xj)
        """
        # hash_key = f'j_{str(j)}_pa_{str(sorted(pa))}'
        hash_key = f'j_{str(j)}_pa_{str(pa)}'

        if self.mdl_cache.__contains__(hash_key):
            return self.mdl_cache[hash_key]

        if len(pa) > 1 and f'j_{str(j)}_pa_{str(pa[:-1])}' not in self.mdl_cache.keys():
            self.score_edge(j, pa[:-1])

        # if len(pa) > 1:
        #     parent_subsets = list(permutations(pa, len(pa)-1))
        #     if not np.any([f'j_{str(j)}_pa_{str(sorted(ss))}' in self.mdl_cache for ss in parent_subsets]):
        #         pass

        bic = fit_edge(pa, j, self.global_params, self.verbosity, self.X, self.model_cache)

        self.mdl_cache[hash_key] = bic

        return bic
