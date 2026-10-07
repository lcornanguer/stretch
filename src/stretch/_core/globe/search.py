import itertools
import time

import numpy as np

from .dag import DAG, is_insignificant
# from score_dags import score_dags
from .upq import UPQ
# from util_dag import gen_dags_from_queue



def dag_exhaustive_search(dag_search: DAG, verbosity=0) -> DAG:
    q = UPQ()
    q = dag_search.initial_edges(q, skip_insignificant=True)
    return _dag_exhaustive_phase(q, dag_search, verbosity)


def dag_tree_search(N, data,
                    is_true_edge= lambda i : lambda j : "",
                    global_params=None,
                    verbosity=0) -> DAG:
    """
    Greedy tree search for causal DAGs
    :param data: data
    :param verbosity: verbosity
    :return: dag
    """
    if verbosity > 0:
        print('\n*** DAG Search ***')

    if global_params is None:
        global_params = {"use_lasso": True, "verbose_iter": False, "init_with_child_centric": True, "max_lag": 6,
                         'pruning_threshold': 0.4, "regression": "poly", "fix_previous_parents": True, "use_bic": False}

    q = UPQ()
    dag_search = DAG(N, data, global_params, is_true_edge, verbosity)
    q = dag_search.initial_edges(q)

    q, dag_search = _dag_forward_phase(q, dag_search, verbosity)
    q, dag_search = _dag_backward_phase(q, dag_search, verbosity)

    return dag_search


def _dag_forward_phase(q: UPQ, dag_search: DAG, verbosity: int) -> (UPQ, DAG):
    st = time.perf_counter()
    if verbosity > 0:
        print('Forward Phase ...')

    while q.pq:
        try:
            pi_edge = q.pop_task()
            node, parent = pi_edge.j, pi_edge.i

            # Check whether adding the edge would result in a cycle
            if dag_search.has_cycle(parent, node):
                continue
            gain, score, pa, score_cur, pa_cur = dag_search.eval_edge_addition(node, parent)

            # Check whether gain is significant
            if is_insignificant(gain):
                continue

            dag_search.add_edge(parent, node, score, gain)  #want true_adj[parent][node] > 0

            # Reconsider children under current model and remove if reversing the edge improves score
            for ch in dag_search._nodes:
                if not dag_search.is_edge(node, ch):
                    continue
                gain = dag_search.eval_edge_flip(node, ch)

                if not is_insignificant(gain):
                    # Remove the edge, update the gain of both edges
                    dag_search.remove_edge(node, ch)
                    edge_fw = dag_search.pair_edges[node][ch]
                    edge_bw = dag_search.pair_edges[ch][node]
                    assert edge_fw.i == node and edge_fw.j == ch
                    assert edge_bw.i == ch and edge_bw.j == node

                    assert not (q.exists_task(edge_fw))  # since this was included in the model, i.e. removed from queue at some point
                    # might have been skipped due to insig: assert (q.exists_task(edge_backward))
                    if (q.exists_task(edge_bw)):
                        q.remove_task(edge_bw)

                    gain_bw, _, _, _, _ = dag_search.eval_edge_addition(edge_bw.i, edge_bw.j)
                    gain_fw, _, _, _, _ = dag_search.eval_edge_addition(edge_fw.i, edge_fw.j)
                    q.add_task(edge_bw, gain_bw * 100)
                    q.add_task(edge_fw, gain_fw * 100)

            # Reconsider edges Xk->Xj in q given the current model as their score changed upon adding Xi->Xj
            for mom in dag_search._nodes:
                # Do not consider Xi,Xj, or current parents/children of Xi
                if node == mom or parent == mom \
                        or dag_search.is_edge(mom, node) or dag_search.is_edge(node, mom):
                    continue
                edge_candidate = dag_search.pair_edges[mom][node]  # pi_dag.init_edges[mom][target]
                gain_mom, score, _, _, _ = dag_search.eval_edge_addition(node, mom)

                if (q.exists_task(edge_candidate)):  #ow. insignificant /skipped
                    q.remove_task(edge_candidate)
                    q.add_task(edge_candidate, gain_mom * 100)
        except (KeyError):  # empty or all remaining entries are tagged as removed
            pass

    if verbosity > 0:
        print(f'Forward: {np.round(time.perf_counter() - st, 2)}s ')
    return q, dag_search


def _dag_backward_phase(q: UPQ, dag_search: DAG, verbosity: int) -> (UPQ, DAG):
    st = time.perf_counter()
    if verbosity > 0:
        print('Backward Phase ...')

    for j in dag_search._nodes:
        parents = dag_search.parents_of(j)
        if len(parents) <= 1:
            continue
        max_gain = -np.inf
        arg_max = None

        # Consider all graphs G' that use a subset of the target's current parents
        min_size = 1  # todo min_size = 0 allowed?
        for k in range(min_size, len(parents)+1):
            parent_sets = itertools.combinations(parents, k)
            for parent_set in parent_sets:
                gain = dag_search.eval_edges(j, list(parent_set))
                if gain > max_gain:
                    max_gain = gain
                    arg_max = parent_set
                #print(f'\tconsidering {parent_set} -> {j}, {np.round(gain[0][0],2)}')
        if (arg_max is not None) and (not is_insignificant(max_gain)):
            if verbosity > 0:
                    print(f'\tupdating {parents} to {arg_max} -> {j}' )
            dag_search.update_edges(j, arg_max)

    if verbosity > 0:
        print(f'Backward: {np.round(time.perf_counter() - st, 2)}s ')

    return q, dag_search
