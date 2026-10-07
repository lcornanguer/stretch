import networkx as nx
from itertools import permutations

from .edge_fitting import fit_edge
from .parent_centric_em import *

class Exhaustive_search:
    def __init__(self, data, verbosity=True, true_dag=None, global_params=None, max_pa=None, allow_cycles=False):
        self.dag = nx.DiGraph()
        self.dag.add_nodes_from(range(len(data)))
        self.model_cache = dict() # will be in learning order unlike in GLOBE or TOPIC
        self.data = data
        self.verbosity = verbosity
        self.true_dag = true_dag
        self.allow_cycles = allow_cycles
        if global_params is None:
            self.global_params = {"use_lasso": True, "verbose_iter": False, "init_with_child_centric": True, 'pruning_threshold': 0.4,
                       "max_lag": 6, "fix_previous_parents": True, "use_bic": False, "regression": "poly"}
        else:
            self.global_params = global_params
        self.max_pa = max_pa if max_pa is not None else len(data.keys())

    def main(self):
        # self.search()
        self.depth_first_search()
        best_dag = self.select_dag()
        return best_dag

    def breadth_first_search(self):
        print(f"Scoring all possible edges...")
        for target in self.dag.nodes:
            for k in range(0, len(self.dag.nodes)):
                if k > self.max_pa: continue
                candidate_parents = [n for n in self.dag.nodes if n != target]
                parent_sets = list(permutations(candidate_parents, k))
                for parent_set in parent_sets:
                    covariates = list(parent_set)
                    self.score_edge(covariates=covariates, target=target)

    def depth_first_search(self):
        for target in self.dag.nodes:
            candidate_parents = [n for n in self.dag.nodes if n != target]
            self._dfs_score(current_parents=[],
                            remaining_parents=candidate_parents,
                            target=target)

    def _dfs_score(self, current_parents, remaining_parents, target):
        # 1. Score current set
        self.score_edge(covariates=current_parents, target=target)

        # 2. Explore children (Permutation logic)
        for i in range(len(remaining_parents)):
            next_parent = remaining_parents[i]

            # New parents list
            new_parents = current_parents + [next_parent]

            if len(new_parents) > self.max_pa: continue

            # REMAINING: Everything EXCEPT the one we just picked
            next_remaining = remaining_parents[:i] + remaining_parents[i + 1:]

            self._dfs_score(new_parents, next_remaining, target)

        # 3. Cleanup
        key = f'j_{str(target)}_pa_{current_parents}'
        if key in self.model_cache:
            target_models = [k for k in self.model_cache.keys() if eval(k.split('_')[1]) == target]
            best_target_score = min([self.model_cache[k]['score'] for k in target_models])
            branch_models = [k for k in target_models if k.split('_')[-1] in [str(current_parents[:i]) for i in range(len(current_parents))]]
            for model in target_models:
                if self.model_cache[model]['score'] > best_target_score and model not in branch_models:
                    self.model_cache[key]['model'] = None

    def generate_all_dags(self):
        nodes = list(self.dag.nodes)
        n = len(nodes)
        possible_edges = [(u, v) for u, v in product(nodes, repeat=2) if u != v]

        all_dags = []

        # Power set of all possible edges
        num_edges = len(possible_edges)
        for i in range(1 << num_edges):
            edges = [possible_edges[j] for j in range(num_edges) if (i >> j) & 1]
            if len(edges) > self.max_pa: continue

            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            if self.allow_cycles or nx.is_directed_acyclic_graph(G):
                all_dags.append(G)

        return all_dags

    def select_dag(self):
        best_score = np.inf
        best_dag = None
        true_dag_score = np.nan
        print(f"Enumerating and scoring candidate DAGs...")
        nodes = list(self.dag.nodes)
        n = len(nodes)
        possible_edges = [(u, v) for u, v in product(nodes, repeat=2) if u != v]

        # Power set of all possible edges
        num_edges = len(possible_edges)
        for i in range(1 << num_edges):

            edges = [possible_edges[j] for j in range(num_edges) if (i >> j) & 1]
            max_pa = max([len([1 for e in edges if e[1] == v]) for v in range(len(nodes))])
            if max_pa > self.max_pa: continue

            G = nx.DiGraph()
            G.add_nodes_from(nodes)
            G.add_edges_from(edges)

            if not nx.is_directed_acyclic_graph(G) and not self.allow_cycles: continue

            # scoring
            dag = G
            score = 0
            instantaneous_edges = list()
            for target in dag.nodes:
                parents = [i for i in dag.predecessors(target)]
                permuts = permutations(parents)
                lowest_score = np.inf
                for permut in permuts:
                    hash_key = f'j_{str(target)}_pa_{list(permut)}'
                    lowest_score = min(lowest_score, self.model_cache[hash_key]['score'])
                    if True in self.model_cache[hash_key]['instantaneous'].values(): instantaneous_edges += [(k, target) for k in self.model_cache[hash_key]['instantaneous']]
                score += lowest_score  # self.model_cache[hash_key]['score']
            if self.true_dag is not None and (nx.to_numpy_array(dag) == nx.to_numpy_array(self.true_dag)).all():
                true_dag_score = score
            instant_cycles = any([(t, p) in instantaneous_edges for (p, t) in instantaneous_edges])
            if score < best_score and not instant_cycles:
                best_score = score
                best_dag = dag
        if self.true_dag is not None:
            print(f'Found best score: {best_score}. True DAG best score: {true_dag_score}')
        return best_dag

    def get_delay_functions(self, dag):
        delay_fcts = {}
        for target in dag.nodes:
            parents = [i for i in dag.predecessors(target)]
            if len(parents) == 0: continue
            permuts = permutations(parents)
            lowest_score = np.inf
            for permut in permuts:
                hash_key = f'j_{str(target)}_pa_{list(permut)}'
                if self.model_cache[hash_key]['score'] < lowest_score:
                    best_hash = hash_key
                    best_permut = permut
                lowest_score = min(lowest_score, self.model_cache[hash_key]['score'])
            for i in range(1, len(best_permut)+1):
                self.score_edge(covariates=list(best_permut[:i]), target=target)
            model = self.model_cache[best_hash]['model']  # has been deleted, we need to retrain it
            path = model.decode(model.Y_eff)
            decoded_dict = model.decode_parent_delays(path)
            g_u = model.recover_parent_centric_g_u(decoded_dict)
            for pa in g_u.keys():
                delay_fcts[(pa, target)] = g_u[pa]
        return delay_fcts

    def score_edge(self,
                  covariates: list,
                  target: int
                   ) -> int:

        hash_key = f'j_{str(target)}_pa_{list(covariates)}'
        if hash_key not in self.model_cache.keys() or self.model_cache[hash_key]['model'] is None:
            bic = fit_edge(covariates, target, self.global_params, self.verbosity, self.data, self.model_cache)
        else:
            bic = self.model_cache[hash_key]['score']

        return bic
