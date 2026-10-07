import math

from .utils import universal_integer_encoding, universal_real_encoding

import numpy as np
from itertools import product

import scipy.stats
from scipy.special import logsumexp
import sklearn
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Lasso, LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold, GroupKFold
from scipy.special import gammaln, gamma
from sklearn.preprocessing import SplineTransformer

import warnings

# Suppress minor division warnings that can occur during initialization/low-probability steps
warnings.filterwarnings("ignore", category=RuntimeWarning)


class FHMM_ParentSet:

    def __init__(self, max_lag=None, max_iter=100, tol=1e-3, verbose=False, parent_names=None, use_lasso=False, pruning_threshold=0, regression='poly', min_lag=0):
        self.verbose = verbose
        self.max_iter = max_iter
        self.tol = tol
        self.pruning = (pruning_threshold > 0)
        self.max_lag = max_lag

        # M: Hidden/Active parent names
        self.parent_names = parent_names
        self.hidden_parent_names = self.parent_names
        self.fixed_parent_names = []
        self.M = len(self.hidden_parent_names)

        # Registry for ALL parents (Hidden and Fixed)
        self.parents = {}
        delays = list(range(min_lag, max_lag + 1))

        self.use_lasso = use_lasso
        self.regression = regression
        self.fct_library_size = len(self.get_nonlinear_library(None)[0])
        self.phi_size = 2 + self.fct_library_size * len(self.parent_names)

        for name in self.parent_names:
            allowed = self.get_allowed_states(delays)
            n_states = len(allowed)
            self.parents[name] = {
                "is_fixed": False,
                "allowed_states": allowed,
                "A": None,
                "A_counts": None,
                "pi": np.ones(n_states) / n_states,
                "phi": np.zeros(self.fct_library_size),  # [linear, quadratic]
                "Z": None, # Will store Z_nodes (dict) or Z_fixed (array)
                "mask": None,
                "initial_n_state": len(allowed) # for state survival probability after pruning
            }
            self.parents[name]['A'] = self.init_transition_matrix(name)

        # Joint state metadata for HIDDEN parents
        self.N_states_m = [len(self.parents[n]["allowed_states"]) for n in self.hidden_parent_names]
        self.joint_state_indices = list(product(*[range(N) for N in self.N_states_m]))
        self.n_states = len(self.joint_state_indices)
        self.idx_to_s_tuple = {i: self.joint_state_indices[i] for i in range(self.n_states)}
        self.s_tuples_arr = np.array([self.idx_to_s_tuple[i] for i in range(self.n_states)])

        # Global Parameters
        self.base_phi = np.zeros(2)  # [Intercept, AR]
        self.Sigma = None

        self.sparsity_pen = 0 # lambda

        # Matrices
        self.A_joint = None #self._calculate_joint_transition_matrix() # compute later to not unnecessarily use fixed states
        for pa in self.parent_names:
            self.parents[pa]["mask"] = (self.parents[pa]["A"] > 0)

        self.Y_eff = None
        self.Y_prev = None
        self.Y = None
        self.X_dict = None

        self.pruning_threshold = pruning_threshold

    def add_fixed_parents(self, fixed_model):
        path_indices = fixed_model.decode(fixed_model.Y_eff)
        lag_diff = self.max_lag - fixed_model.max_lag
        aligned_path = path_indices[lag_diff:]

        # Identify hidden names in the donor model to extract the correct tuple indices
        fixed_model_hidden_names = [n for n, d in fixed_model.parents.items() if not d.get("is_fixed")]
        name_to_hidden_idx = {name: i for i, name in enumerate(fixed_model_hidden_names)}

        # Determine width based on chosen method
        col_step = self.fct_library_size

        for name in fixed_model.parent_names:
            self.fixed_parent_names.append(name)
            old_parent_data = fixed_model.parents[name].copy()

            # 1. Trace the state sequence
            if name in name_to_hidden_idx:
                h_idx = name_to_hidden_idx[name]
                parent_state_seq = [fixed_model.idx_to_s_tuple[p][h_idx] for p in aligned_path]

                # 2. Calculate LL for this parent's path
                A = old_parent_data["A"]
                pi = old_parent_data["pi"]

                # Initial state prob
                current_fixed_ll = np.log2(pi[parent_state_seq[0]] + 1e-12)

                # Transition probs
                from_states = parent_state_seq[:-1]
                to_states = parent_state_seq[1:]
                current_fixed_ll += np.sum(np.log2(A[from_states, to_states] + 1e-12))

                # 3. Accumulate and store
                old_parent_data["fixed_ll"] = current_fixed_ll
                old_parent_data["fixed_state_indices"] = parent_state_seq  # keep for future donor steps

            else:
                parent_state_seq = None

            T_eff = len(aligned_path)
            Z_fixed_array = np.zeros((T_eff, col_step))

            # 2. Reconstruct the Fixed Path Array
            if name in name_to_hidden_idx:
                # Transfer from hidden state registry to a fixed path array
                old_Z_nodes = fixed_model.parents[name]["Z"]
                for t, s_idx in enumerate(parent_state_seq):
                    # Slice the block [t, 0:col_step]
                    Z_fixed_array[t, :] = old_Z_nodes[s_idx][t + lag_diff, :]
            else:
                # Simply slice the existing fixed path from the donor model
                Z_fixed_array = old_parent_data["Z_fixed_path"][lag_diff:]

            # 3. Update Registry
            self.parents[name] = old_parent_data
            self.parents[name]["is_fixed"] = True
            self.parents[name]["Z_fixed_path"] = Z_fixed_array

        # 4. Refresh Metadata
        self.update_ols_metadata()
        # Note: update_ols_metadata should recalculate self.phi_size based on col_step

        hidden_names = [n for n, d in self.parents.items() if not d.get("is_fixed")]
        self.N_states_m = [len(self.parents[n]["allowed_states"]) for n in hidden_names]
        self.M = len(hidden_names)
        self.joint_state_indices = list(product(*[range(N) for N in self.N_states_m]))
        self.n_states = len(self.joint_state_indices)
        self.idx_to_s_tuple = {i: self.joint_state_indices[i] for i in range(self.n_states)}

    def update_ols_metadata(self):
        """
        Synchronizes the OLS column indices with the current state of the registry.
        This handles both Hidden and Fixed parents.
        """

        # 1. Identify which parents are fixed and which are hidden
        # We explicitly pull fixed parents first to match our OLS assembly logic
        fixed_names = [n for n, d in self.parents.items() if d.get("is_fixed", False)]
        hidden_names = [n for n, d in self.parents.items() if not d.get("is_fixed", False)]

        self.hidden_parent_names = hidden_names  # This is the list for hidden state logic
        self.M = len(hidden_names)  # This is the count of hidden chains

        # 2. Store the order for global use
        self.ols_variable_order = fixed_names + hidden_names

        # 3. Calculate total size:
        # [Intercept(1), AR(1), 2 cols per fixed parent, 2 cols per hidden parent]
        self.phi_size = 2 + (self.fct_library_size * len(self.ols_variable_order))
        self.hidden_phi_start = 2 + (self.fct_library_size * len(fixed_names))

        if self.verbose:
            print(f"OLS Metadata Updated: {len(fixed_names)} fixed, {len(hidden_names)} hidden.")
            print(f"Total Phi size: {self.phi_size}")

    def init_transition_matrix(self, name):
        """
        Initializes M independent transition matrices A_m, respecting the constraints
        within each parent's allowed state set.
        """
        allowed_states_m = self.parents[name]['allowed_states']
        N_m = len(allowed_states_m)

        A_m = np.zeros((N_m, N_m))

        for i, s1 in enumerate(allowed_states_m):
            for j, s2 in enumerate(allowed_states_m):
                valid = True

                if s1[0] == 1 and np.sum(s2) == 0: continue # from max lag, the lag cannot be increased

                if len(np.where(s1 == 1)[0]) == 0 and len(np.where(s2 == 1)[0]) == 0:
                    A_m[i, j] = 0
                    continue

                if len(np.where(s1 == 1)[0]) == 0 or len(np.where(s2 == 1)[0]) == 0:
                    A_m[i, j] = 1.0
                    continue

                if len(np.where(s1 == 1)[0]) > 0:
                    last_1 = np.where(s1 == 1)[0][-1]
                    last_2 = np.where(s2 == 1)[0][-1]
                    if last_1 > last_2:
                        valid = False
                    if len(np.where(s2[:last_1] == 1)[0]) > 0:
                        valid = False
                    elif last_1 < last_2:
                        if len(np.where(s2[last_1:last_2 + 1] == 0)[0]) > 0:
                            valid = False

                if valid:
                    A_m[i, j] = 1.0

            if np.sum(A_m[i, :]) > 0:
                A_m[i, :] /= np.sum(A_m[i, :])
            else:
                A_m[i, :] = 1.0 / N_m

        return A_m

    def _calculate_joint_transition_matrix(self):
        A_joint = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            s_i_tuple = self.idx_to_s_tuple[i]
            for j in range(self.n_states):
                s_j_tuple = self.idx_to_s_tuple[j]

                joint_prob = 1.0
                # Track index 'm_hidden' to map correctly to s_tuple
                m_hidden = 0
                for name, data in self.parents.items():
                    if data.get("is_fixed"):
                        continue  # Skip fixed parents

                    # Multiply only hidden transitions
                    joint_prob *= data["A"][s_i_tuple[m_hidden], s_j_tuple[m_hidden]]
                    m_hidden += 1

                A_joint[i, j] = joint_prob
        # return A_joint

        # Normalize rows
        row_sums = A_joint.sum(axis=1, keepdims=True)
        A_joint = np.divide(A_joint, row_sums, out=np.zeros_like(A_joint), where=row_sums != 0)
        return A_joint

    def _emission_prob_all_from_Z_nodes(self, Y_eff):
        T_eff = len(Y_eff)
        N_joint = self.n_states

        # 1. Base Dynamics: Intercept + AR(1)
        # Shape: (T_eff, 1)
        mu_base = (self.base_phi[0] + self.base_phi[1] * self.Y_prev)[:, None]

        # 2. Add Fixed Parents Contribution (Vectorized for Poly/RBF)
        for name, data in self.parents.items():
            if data.get("is_fixed"):
                phi = data["phi"]
                Z_fixed = data["Z_fixed_path"]
                # Dot product handles any number of columns (2 for poly, K for rbf)
                mu_base += (Z_fixed @ phi)[:, None]

        # 3. Precompute Hidden Parent contributions
        parent_mu_hidden = []
        for name in self.hidden_parent_names:
            data = self.parents[name]
            phi = data["phi"] # np.zeros(data["phi"].shape)
            Z_nodes = data["Z"]  # Dict of {s_m: (T_eff, K)}

            # Each hidden state for this parent generates a T_eff signal
            # mu_m shape: (N_states_m, T_eff)
            mu_m = np.array([Z_nodes[s_m] @ phi for s_m in range(len(Z_nodes))])
            parent_mu_hidden.append(mu_m)

        # 4. Create the Joint Mean Matrix: (T_eff, N_joint)
        # We start with the base signal and add specific parent contributions
        mu = mu_base + np.zeros((T_eff, N_joint))

        # 5. Broadcast hidden contributions based on the joint state mapping
        # s_tuples[:, m] gives the state of hidden parent m for every joint state i
        s_tuples = np.array([self.idx_to_s_tuple[i] for i in range(N_joint)])
        for m in range(self.M):
            # parent_mu_hidden[m][...] selects rows based on parent states
            # .T aligns it to (T_eff, N_joint)
            mu += parent_mu_hidden[m][s_tuples[:, m]].T

        # 6. Gaussian Likelihood Calculation
        sigma = max(self.Sigma.item(), 1e-9)
        residual = Y_eff[:, None] - mu
        residual = -abs(residual)

        # Calculate in log-space first to avoid precision issues with small sigma
        log_E = -0.5 * np.log2(2 * np.pi * sigma) - 0.5 * (residual ** 2 / sigma) / np.log(2) # sigma is the residual variance, not the std

        # Clip log_E to prevent exp(small) = 0
        E = np.exp2(np.clip(log_E, -700, None))
        E[E < 1e-300] = 1e-300

        return E

    def get_ols_mapping(self):
        """
        Returns the exact order of parents as they appear in the OLS matrix.
        Logic: [Fixed_Parents, Hidden_Parents]
        """
        fixed = [n for n, d in self.parents.items() if d.get("is_fixed", False)]
        hidden = self.hidden_parent_names  # Order defined during __init__
        return fixed, hidden

    def scatter_phi(self, flat_phi):
        """ Distributes the flat Phi vector back into the parent registry. """
        self.base_phi = flat_phi[0:2]
        fixed_names, hidden_names = self.get_ols_mapping()

        order = fixed_names + hidden_names
        curr = 2
        for name in order:
            self.parents[name]["phi"] = flat_phi[curr: curr + self.fct_library_size]
            curr += self.fct_library_size

    def _initialize_parameters(self, Y, X_dict):
        self.update_ols_metadata()
        fixed_names, hidden_names = self.get_ols_mapping()

        self.Y_eff = Y[self.max_lag:]
        self.Y_prev = Y[self.max_lag - 1: -1]
        T_eff = len(self.Y_eff)

        # 1. Precalculate Z with the chosen regression method
        # Pass n_centers only if using RBF
        # Z_hidden = self._precalculate_Z_per_node(X_dict, Y)
        self._precalculate_Z_per_node(X_dict, Y)
        col_step = self.fct_library_size

        # for name in hidden_names:
        #     self.parents[name]["Z"] = Z_hidden[name]

        # 2. Build OLS Design Matrix
        X_OLS = np.zeros((T_eff, self.phi_size))
        if self.use_lasso: # todo: to keep?
            X_OLS[:, 0] = 1.0  # Keep it for Polynomial mode
        else:
            X_OLS[:, 0] = 1.0  # Keep it for Polynomial mode
        X_OLS[:, 1] = self.Y_prev

        curr_col = 2
        warmup_vec = np.array([0] * self.max_lag + [1])

        # FIRST: Fill Fixed Parents
        for name in fixed_names:
            # Note: Fixed paths must have been pre-calculated with the correct n_centers
            X_OLS[:, curr_col: curr_col + col_step] = self.parents[name]["Z_fixed_path"]
            curr_col += col_step

        # SECOND: Fill Hidden Parents
        for name in hidden_names:
            parent = self.parents[name]
            try:
                s_idx = [np.array_equal(v, warmup_vec) for v in parent["allowed_states"]].index(True)
            except ValueError:
                s_idx = 0
                for i, v in enumerate(parent["allowed_states"]):
                    if not np.all(v == 0):
                        s_idx = i
                        break

            # Dynamically fill based on col_step
            X_OLS[:, curr_col: curr_col + col_step] = parent["Z"][s_idx]
            curr_col += col_step

        # 3. Fit OLS
        Phi_init, _, _, _ = np.linalg.lstsq(X_OLS, self.Y_eff, rcond=1e-10)
        self.scatter_phi(Phi_init)

        # 4. Sigma (using T_eff - phi_size as denominator)
        SSE = np.sum((self.Y_eff - X_OLS @ Phi_init) ** 2)
        df = T_eff - self.phi_size
        self.Sigma = np.array([max(SSE / df, 1e-9)]) if df > 0 else np.array([np.var(Y) / 2.0])

        # 5. Joint Transitions
        self.A_joint = self._calculate_joint_transition_matrix()

    def _precalculate_Z_per_node(self, X_dict, Y):
        """
        """
        T = len(Y)
        L_global = self.max_lag
        T_eff = T - L_global

        for name in self.hidden_parent_names:
            X = X_dict[name]
            allowed_states = self.parents[name]["allowed_states"]
            parent_lag_len = len(allowed_states[0])

            # 1. Prepare Windowed Data (Shape: parent_lag_len, T_eff)
            X_windows = np.zeros((parent_lag_len, T_eff))
            for lag in range(parent_lag_len):
                start = lag
                end = T - (L_global - lag)
                X_windows[lag, :] = X[start:end]

            X_raw = X_dict[name]

            # 1. Transform raw X to Nonlinear Library
            # Shape: (T, lib_size) where lib_size is 6 (1 linear + 5 RBF)
            X_lib = self.get_nonlinear_library(X_raw)
            lib_size = X_lib.shape[1]

            # 2. Create the "Delayed Cube"
            # We need a shape: (delay, T_eff, lib_size)
            allowed_states = self.parents[name]["allowed_states"]
            parent_lag_len = len(allowed_states[0])

            X_cube = np.zeros((parent_lag_len, T_eff, lib_size))
            for lag in range(parent_lag_len):
                start = lag
                end = T - (L_global - lag)
                X_cube[lag, :, :] = X_lib[start:end, :]

            node_data = {}

            # 3. Apply state-specific masks
            for s_idx, state_vec in enumerate(allowed_states):
                # state_vec shape: (parent_lag_len,)
                # We broadcast it to (parent_lag_len, 1, 1) to multiply against the cube
                mask = np.array(state_vec).reshape(parent_lag_len, 1, 1)

                # Sum across the lag dimension
                # Result shape: (T_eff, lib_size)
                node_data[s_idx] = np.sum(X_cube * mask, axis=0)

            self.parents[name]["Z"] = node_data

    def _forward_backward(self, Y):
        T_eff = len(self.Y_eff)
        N_joint = self.n_states

        # 1. pi_joint: ONLY iterate over hidden parents (self.parent_names)
        # Calculate pi_joint
        pi_joint = np.zeros(self.n_states)
        for i in range(self.n_states):
            s_tuple = self.idx_to_s_tuple[i]
            prob = 1.0
            m_hidden = 0
            for name, data in self.parents.items():
                if data.get("is_fixed"):
                    continue
                prob *= data["pi"][s_tuple[m_hidden]]
                m_hidden += 1
            pi_joint[i] = prob

        # pi_joint /= (np.sum(pi_joint) + 1e-12)
        denom = np.sum(pi_joint)
        if denom > 1e-15:
            pi_joint /= denom
        else:
            pi_joint = np.ones_like(pi_joint) / len(pi_joint) # If the probability is effectively zero, reset to uniform

        # Normalize safely
        denom_pi = np.sum(pi_joint)
        pi_joint = pi_joint / denom_pi if denom_pi > 0 else np.ones(N_joint) / N_joint

        # 2. PRE-CALCULATE EMISSION PROBABILITIES (E_t)
        # Now calls the parameter-free version that uses the registry internally
        E = self._emission_prob_all_from_Z_nodes(self.Y_eff)

        # 3. FORWARD PASS (Alpha)
        alpha = np.zeros((T_eff, N_joint))
        alpha[0, :] = pi_joint * E[0, :]
        scale = np.sum(alpha[0, :])
        if scale == 0:
            alpha[0, :] = 1.0 / N_joint
            scale = 1.0

        alpha[0, :] /= scale
        log_likelihood = np.log2(scale)

        for t in range(1, T_eff):
            # Prediction step using the pre-calculated A_joint
            B_curr = alpha[t - 1, :].reshape(self.N_states_m)
            hidden_names = [n for n, d in self.parents.items() if not d.get("is_fixed")]

            for m, name in enumerate(hidden_names):
                A_m = self.parents[name]["A"]
                B_curr = np.moveaxis(B_curr, m, 0)
                B_curr = np.einsum('i...,ij->j...', B_curr, A_m)
                B_curr = np.moveaxis(B_curr, 0, m)

            alpha[t, :] = B_curr.flatten() * E[t, :]

            scale = np.sum(alpha[t, :])
            if scale == 0:
                # Numerical fallback: keep previous distribution
                alpha[t, :] = alpha[t - 1, :]
            else:
                alpha[t, :] /= scale
                log_likelihood += np.log2(scale)

        # 4. BACKWARD PASS (Beta)
        beta = np.zeros((T_eff, N_joint))
        beta[T_eff - 1, :] = 1.0
        for t in range(T_eff - 2, -1, -1):
            # Beta step: A_joint @ (Emission * Beta_next)
            beta[t, :] = self.A_joint @ (E[t + 1, :] * beta[t + 1, :])
            b_scale = np.sum(beta[t, :])
            if b_scale > 0:
                beta[t, :] /= b_scale
            else:
                beta[t, :] = 1.0 / N_joint

        # 5. COMPUTE SMOOTHED PROBABILITIES (Gamma and Xi)
        # Gamma: P(q_t | Y)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # Xi: P(q_t, q_{t+1} | Y)
        xi = np.zeros((T_eff - 1, N_joint, N_joint))
        for t in range(T_eff - 1):
            E_beta_term = E[t + 1, :] * beta[t + 1, :]
            # Outer product of alpha and (E*beta) weighted by transitions
            numerator = alpha[t, :].reshape(-1, 1) * self.A_joint * E_beta_term.reshape(1, -1)
            denom = np.sum(numerator)

            if denom > 1e-12:
                xi[t, :, :] = numerator / denom
            else:
                # Fallback to transition matrix if observations are uninformative
                xi[t, :, :] = self.A_joint / self.A_joint.sum()

        return log_likelihood, gamma, xi

    def _maximization_step(self, Y, gamma, xi):
        Y_eff = Y[self.max_lag:]
        T_eff = len(self.Y_eff)

        # 1. ORCHESTRATE VARIABLE ORDERING
        fixed_names = [n for n, d in self.parents.items() if d.get("is_fixed")]
        hidden_names = [n for n, d in self.parents.items() if not d.get("is_fixed")]

        # DYNAMIC OFFSET: Determine how many columns each parent takes
        col_step = self.fct_library_size

        # 2. DATA STACKING (The "Long" Matrix)
        Y_long = np.tile(Y_eff, self.n_states)
        W_long = gamma.T.flatten()
        Z_long = np.zeros((T_eff * self.n_states, self.phi_size))

        Z_const = np.zeros((T_eff, self.phi_size))
        Z_const[:, 0] = 1.0
        Z_const[:, 1] = self.Y_prev

        curr_col = 2
        for name in fixed_names:
            Z_const[:, curr_col: curr_col + col_step] = self.parents[name]["Z_fixed_path"]
            curr_col += col_step

        hidden_col_start = curr_col
        for i in range(self.n_states):
            Z_i = Z_const.copy()
            s_tuple = self.idx_to_s_tuple[i]
            h_col = hidden_col_start
            for m, name in enumerate(hidden_names):
                Z_i[:, h_col: h_col + col_step] = self.parents[name]["Z"][s_tuple[m]]
                h_col += col_step
            Z_long[i * T_eff: (i + 1) * T_eff, :] = Z_i

        # 3. SCALING (Only for Parents/AR)
        # Indices 1 onwards (Column 0 is Intercept)
        parent_indices = np.arange(1, self.phi_size)
        W_sum = np.sum(W_long)
        mu = np.sum(W_long[:, np.newaxis] * Z_long[:, parent_indices], axis=0) / W_sum
        sigma = np.sqrt(np.sum(W_long[:, np.newaxis] * (Z_long[:, parent_indices] - mu) ** 2, axis=0) / W_sum)
        sigma[sigma < 1e-9] = 1.0

        Z_scaled = Z_long.copy()
        Z_scaled[:, parent_indices] = (Z_scaled[:, parent_indices] - mu) / sigma

        # 4. SOLVER CALL
        # Pass groups for GroupKFold consistency
        groups = np.tile(np.arange(T_eff), self.n_states)

        if self.use_lasso:
            sklearn.set_config(enable_metadata_routing=True)
            cv = GroupKFold(n_splits=3) #, random_state=np.random.default_rng(42)) # Setting a random_state has no effect since shuffle is False
            model = LassoCV(
                alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 2],
                fit_intercept=False,
                max_iter=10000,
                tol=0.01,
                cv=cv,
                random_state=42
            )
            model.fit(Z_scaled, Y_long, sample_weight=W_long, groups=groups)
            self.sparsity_pen = model.alpha_
            beta_scaled = model.coef_

        else:
            model = LinearRegression(fit_intercept=False)
            model.fit(Z_scaled, Y_long, sample_weight=W_long)
            self.sparsity_pen = 0
            beta_scaled = model.coef_

        # 5. BACK-TRANSFORMATION
        Phi_new = beta_scaled.copy()
        Phi_new[parent_indices] = beta_scaled[parent_indices] / sigma
        Phi_new[0] -= np.sum((beta_scaled[parent_indices] / sigma) * mu)
        # Phi_new[2:] = np.zeros(self.phi_size-2)

        self.scatter_phi(Phi_new)


        #
        #
        #
        #
        #
        #
        #
        #
        #
        # if self.use_lasso:
        #     # 2. UPDATE OBSERVATION MODEL (Weighted LASSO)
        #     # We stack all possible joint-state paths to create one massive weighted regression
        #     Y_long = np.tile(Y_eff, self.n_states)
        #     W_long = gamma.T.flatten()  # Responsibilities for every sample-state pair
        #     Z_long = np.zeros((T_eff * self.n_states, self.phi_size))
        #
        #     # Base Regressors: Intercept, AR(1)
        #     Z_const = np.zeros((T_eff, self.phi_size))
        #     Z_const[:, 0] = 1.0
        #     Z_const[:, 1] = self.Y_prev
        #
        #     # Add Fixed Parents using dynamic offset
        #     curr_col = 2
        #     for name in fixed_names:
        #         X_fixed = self.parents[name]["Z_fixed_path"]
        #         Z_const[:, curr_col: curr_col + col_step] = X_fixed
        #         curr_col += col_step
        #
        #     hidden_col_start = curr_col
        #
        #     for i in range(self.n_states):
        #         Z_i = Z_const.copy()
        #         s_tuple = self.idx_to_s_tuple[i]
        #         h_col = hidden_col_start
        #         for m, name in enumerate(hidden_names):
        #             Z_i[:, h_col: h_col + col_step] = self.parents[name]["Z"][s_tuple[m]]
        #             h_col += col_step
        #
        #         # Place in the long design matrix
        #         row_start, row_end = i * T_eff, (i + 1) * T_eff
        #         Z_long[row_start:row_end, :] = Z_i
        #
        #     Z_all = Z_long.copy()
        #
        #     # 2. Scale ONLY the parents (index 2 onwards)
        #     # Leave Column 0 (Intercept) and Column 1 (AR) in raw units
        #     parent_indices = np.arange(1, self.phi_size)
        #     W_sum = np.sum(W_long)
        #     mu = np.sum(W_long[:, np.newaxis] * Z_all[:, parent_indices], axis=0) / W_sum
        #     sigma = np.sqrt(np.sum(W_long[:, np.newaxis] * (Z_all[:, parent_indices] - mu) ** 2, axis=0) / W_sum)
        #     sigma[sigma < 1e-9] = 1.0
        #
        #     Z_all[:, parent_indices] = (Z_all[:, parent_indices] - mu) / sigma
        #
        #     sklearn.set_config(enable_metadata_routing=True)
        #     groups = np.tile(np.arange(T_eff), self.n_states)
        #     cv = GroupKFold(n_splits=3) # cv = KFold(n_splits=3, shuffle=True, random_state=42)
        #     lasso = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2], fit_intercept=False, max_iter=10000, tol=0.01, cv=cv) # alpha is lambda
        #     # lasso = RidgeCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 5, 10], fit_intercept=False, cv=cv) # alpha is lambda
        #     lasso.fit(Z_all, Y_long, sample_weight=W_long, groups=groups)
        #     # print(f'LASSO lambda: {lasso.alpha_}')
        #     beta_all = lasso.coef_
        #     self.sparsity_pen = lasso.alpha_
        #
        #     # # Unpenalized version
        #     # ols = LinearRegression(fit_intercept=False)
        #     # ols.fit(Z_all, Y_long, sample_weight=W_long)
        #     # beta_all = ols.coef_
        #     # self.sparsity_pen = 0
        #     # # End unpenalized version
        #
        #
        #     # 4. Back-transform parents
        #     Phi_new = beta_all.copy()
        #     Phi_new[parent_indices] = beta_all[parent_indices] / sigma
        #     # Adjust intercept for the parent mean-shifts
        #     Phi_new[0] -= np.sum((beta_all[parent_indices] / sigma) * mu)
        #
        #     self.scatter_phi(Phi_new)
        #
        #     # # 1. 'Protect' the Intercept and AR(1)
        #     # # Fit a simple weighted OLS on ONLY the first two columns (Intercept, AR)
        #     # protected_indices = [0, 1]
        #     # parent_indices = np.arange(2, self.phi_size)
        #     #
        #     # ols = LinearRegression(fit_intercept=False)
        #     # ols.fit(Z_long[:, protected_indices], Y_long, sample_weight=W_long)
        #     #
        #     # # 2. Get the Residuals (what the AR model couldn't explain)
        #     # Y_resid = Y_long - ols.predict(Z_long[:, protected_indices])
        #     #
        #     # To give all fcts the same importance (because their scale is not identical)
        #     # 1. Identify features to scale (skip column 0 if it's your constant intercept)
        #     # We don't scale the intercept because its "variance" is 0.
        #     # features_to_scale = parent_indices
        #     #
        #     # 2. Compute Weighted Mean and Std Dev
        #     # We use the weights from the E-step (gamma) to find the effective scale
        #     # W_sum = np.sum(W_long)
        #     # mu = np.sum(W_long[:, np.newaxis] * Z_long[:, features_to_scale], axis=0) / W_sum
        #     # # Weighted variance
        #     # var = np.sum(W_long[:, np.newaxis] * (Z_long[:, features_to_scale] - mu) ** 2, axis=0) / W_sum
        #     # sigma = np.sqrt(var)
        #     # sigma[sigma == 0] = 1.0  # Avoid division by zero for constant features
        #     #
        #     # # 3. Scale the Matrix
        #     # Z_scaled = (Z_long[:, features_to_scale] - mu) / sigma
        #     #
        #     # # 4. Fit LASSO
        #     # sklearn.set_config(enable_metadata_routing=True)
        #     # groups = np.tile(np.arange(T_eff), self.n_states)
        #     # cv = GroupKFold(n_splits=3)
        #     # # lasso = Lasso(alpha=0, fit_intercept=False, max_iter=10000) # alpha is lambda
        #     # lasso = LassoCV(alphas=[0.00001, 0.0001, 0.001, 0.01, 0.1], fit_intercept=False, max_iter=10000, cv=cv) # alpha is lambda
        #     # lasso.fit(Z_scaled, Y_resid, sample_weight=W_long, groups=groups)
        #     # print(f'lasso lambda: {lasso.alpha_}')
        #     #
        #     # beta_parents_scaled = lasso.coef_
        #     # self.sparsity_pen = lasso.alpha_
        #     #
        #     # Phi_new = np.zeros(self.phi_size)
        #     # Phi_new[parent_indices] = beta_parents_scaled / sigma
        #     # Phi_new[protected_indices] = ols.coef_
        #     # # CRITICAL: The intercept (index 0) must absorb the mean-shifts from the scaled parents
        #     # # Because: beta * (Z - mu)/sigma  =>  (beta/sigma)*Z - (beta*mu/sigma)
        #     # mean_shift = np.sum(beta_parents_scaled * mu / sigma)
        #     # Phi_new[0] -= mean_shift
        #     # self.scatter_phi(Phi_new)
        #
        # else:
        #     # 2. UPDATE OBSERVATION MODEL (Phi and Sigma)
        #     Total_Covariance = np.zeros((self.phi_size, self.phi_size))
        #     Total_Cross_Covariance = np.zeros(self.phi_size)
        #
        #     # Base Regressors: Intercept, AR(1)
        #     Z_const = np.zeros((T_eff, self.phi_size))
        #     Z_const[:, 0] = 1.0
        #     Z_const[:, 1] = self.Y_prev
        #
        #     # Add Fixed Parents using dynamic offset
        #     curr_col = 2
        #     for name in fixed_names:
        #         X_fixed = self.parents[name]["Z_fixed_path"]
        #         Z_const[:, curr_col: curr_col + col_step] = X_fixed
        #         curr_col += col_step
        #
        #     hidden_col_start = curr_col
        #
        #     # Accumulate statistics over all joint states
        #     for i in range(self.n_states):
        #         Z_i = Z_const.copy()
        #         s_tuple = self.idx_to_s_tuple[i]
        #
        #         h_col = hidden_col_start
        #         for m, name in enumerate(hidden_names):
        #             s_m = s_tuple[m]
        #             # Grab the block of features for this specific state
        #             Z_i[:, h_col: h_col + col_step] = self.parents[name]["Z"][s_m]
        #             h_col += col_step
        #
        #         w = gamma[:, i]
        #         W_Z_i = Z_i * w[:, np.newaxis]
        #         Total_Covariance += W_Z_i.T @ Z_i
        #         Total_Cross_Covariance += W_Z_i.T @ Y_eff
        #
        #     # Solve for Phi
        #     Total_Covariance += 1e-6 * np.eye(self.phi_size)
        #     Phi_new = np.linalg.solve(Total_Covariance, Total_Cross_Covariance)
        #     self.scatter_phi(Phi_new)

        # Update Sigma (Vectorized calculation)
        SSE = 0
        for i in range(self.n_states):
            Z_i = Z_const.copy()
            s_tuple = self.idx_to_s_tuple[i]
            h_col = hidden_col_start
            for m, name in enumerate(hidden_names):
                Z_i[:, h_col: h_col + col_step] = self.parents[name]["Z"][s_tuple[m]]
                h_col += col_step

            residuals = Y_eff - Z_i @ Phi_new
            SSE += np.sum(gamma[:, i] * (residuals ** 2))

        self.Sigma = np.array([max(SSE / T_eff, 1e-6)])

        # 3. UPDATE TRANSITION MODEL (Same logic as before)
        xi_summed = np.sum(xi, axis=0)
        new_shape = tuple(self.N_states_m) + tuple(self.N_states_m)
        xi_tensor = xi_summed.reshape(new_shape)
        num_hidden = len(hidden_names)

        for m, name in enumerate(hidden_names):
            parent = self.parents[name]
            axes_to_sum = [k for k in range(2 * num_hidden) if k != m and k != (num_hidden + m)]
            xi_m = np.sum(xi_tensor, axis=tuple(axes_to_sum))

            transition_prior = 1.0
            # mask = (parent["A"] > 0)
            mask = parent["mask"]
            A_new = (xi_m + transition_prior) * mask
            parent['A_counts'] = A_new

            row_sums = A_new.sum(axis=1, keepdims=True)
            parent["A"] = np.divide(A_new, row_sums,
                                    out=np.ones_like(A_new) / len(parent["allowed_states"]),
                                    where=row_sums != 0)

            gamma_m_sum = np.zeros(len(parent["allowed_states"]))
            for i in range(self.n_states):
                s_m = self.idx_to_s_tuple[i][m]
                gamma_m_sum[s_m] += gamma[0, i]

            # parent["pi"] = gamma_m_sum / (np.sum(gamma_m_sum) + 1e-12)
            # Sum once to save compute
            denom = np.sum(parent["pi"])
            if denom > 1e-15:
                parent["pi"] /= denom
            else:
                parent["pi"] = np.ones_like(parent["pi"]) / len(parent["pi"]) # If the probability is effectively zero, reset to uniform

        # 4. Finalize
        self.A_joint = self._calculate_joint_transition_matrix()

    def _init_transition_prob_with_state_sequences(self, state_sequences):
        """
        Initializes transition matrices (A) for HIDDEN parents using provided state sequences.
        state_sequences: Dict {parent_name: list_of_bit_vectors}
        """

        def find_index(state_vec, allowed_list):
            for idx, allowed_vec in enumerate(allowed_list):
                if np.array_equal(state_vec, allowed_vec):
                    return idx
            raise ValueError(f"State vector not found in allowed states.")

        # Iterate through the registry
        for name, data in self.parents.items():
            # Skip fixed parents: they don't participate in the HMM learning process
            if data.get("is_fixed") or name not in state_sequences:
                continue

            # 1. Map bit-vectors to state indices
            seq = state_sequences[name]
            allowed = data["allowed_states"]
            state_indices = [find_index(s, allowed) for s in seq]

            # 2. Initialize Count Matrix
            N_m = len(allowed)
            # Small prior (1e-4) ensures no transition is mathematically impossible unless masked
            A = np.full((N_m, N_m), fill_value=1e-4)

            # 3. Count transitions
            for t in range(1, len(state_indices)):
                s_from = state_indices[t - 1]
                s_to = state_indices[t]
                A[s_from, s_to] += 1

            # 4. Enforce structural constraints
            # Use the mask stored in the registry for this specific parent
            if "mask" in data:
                A *= data["mask"]

            # 5. Row-normalize to get probabilities
            row_sums = A.sum(axis=1, keepdims=True)
            # If a row sum is 0 (due to masking), fallback to uniform distribution
            data["A"] = np.divide(A, row_sums, out=np.ones_like(A) / N_m, where=row_sums != 0)

        # 6. Recompute the joint transition matrix for the remaining hidden state space
        self.A_joint = self._calculate_joint_transition_matrix()

    def fit(self, X_dict, Y, warm_init=None):
        """ The input data should come in a dict with the variable name as key and an array as value. """
        # Warm init
        self.X_dict, self.Y = X_dict, Y
        if warm_init is not None:
            self._init_transition_prob_with_state_sequences(warm_init)

        T = len(Y)
        T_eff = T - self.max_lag

        # if len(X_list) != self.M:
        #     raise ValueError(f"Input X_list has {len(X_list)} covariates, but model expected {self.M}.")
        if T_eff <= 0:
            raise ValueError(f"Time series length ({T}) is too short. Need at least {self.max_lag + 1} observations.")

        # Initialize parameters, including OLS warm-up
        if isinstance(Y, list): Y = np.array(Y)
        self._initialize_parameters(Y, X_dict)

        # 2. PRE-CALCULATE ALL REGRESSOR DATA ONCE
        self._precalculate_Z_per_node(X_dict, Y)

        prev_logL = -np.inf
        log_likelihood_history = []

        if self.verbose:
            print(f"Starting EM optimization for parent-centric FHMM with {self.M} chains...")
            print(f"Total joint states: {self.n_states}")

        for it in range(self.max_iter):
            logL, gamma, xi = self._forward_backward(Y) # E-step
            self._maximization_step(Y, gamma, xi) # M-step

            all_zeroed = np.any([sum([c != 0 for c in pa_info['phi']]) == 0 for pa_info in self.parents.values()])
            if all_zeroed: print(f'At least one parent has all regression coefficients are equal to 0!')

            log_likelihood_history.append(logL)

            if self.verbose and (it%10 == 0):
                print(f"Iteration {it + 1}, LogL={logL:.4f}")
            if np.abs(logL - prev_logL) < self.tol:
                if self.verbose: print(f"Convergence reached at iteration {it + 1}. LogL={log_likelihood_history[-1]:.4f}")
                break
            prev_logL = logL

        # Pruning and re-training
        if self.pruning:
            pruned = self.prune(gamma)
            if pruned:
                if self.verbose:
                    print(f"Restarting EM optimization after pruning...")
                # Re-calculate Z_node and A_joint because state space N_joint and A_m have changed
                self._precalculate_Z_per_node(X_dict, Y)

                for it in range(self.max_iter):
                    logL, gamma, xi = self._forward_backward(Y)  # E-step
                    self._maximization_step(Y, gamma, xi)  # M-step

                    log_likelihood_history.append(logL)

                    if self.verbose and it%10 == 0:
                        print(f"Iteration {it + 1}, LogL={logL:.4f}")
                    if np.abs(logL - prev_logL) < self.tol:
                        if self.verbose: print(f"Convergence reached at iteration {it + 1}. LogL={log_likelihood_history[-1]:.4f}")
                        break
                    prev_logL = logL

        finalLL = log_likelihood_history[-1]
        for p in self.fixed_parent_names:
            finalLL += self.parents[p]["fixed_ll"]

        all_zeroed = np.any([sum([c != 0 for c in pa_info['phi']]) == 0 for pa_info in self.parents.values()])
        if all_zeroed: return - np.inf

        return finalLL #log_likelihood_history[-1]

    def get_allowed_states(self, delays):
        mini = min(delays)
        maxi = max(delays) + 1

        # Use a set of tuples to track unique patterns
        unique_states = set()

        # Add the initial empty state
        empty_state = np.zeros(self.max_lag+1)
        unique_states.add(tuple(empty_state))

        for start in range(mini, maxi):
            for end in range(start, maxi):
                state = np.zeros(self.max_lag+1)
                state[start:end + 1] = 1
                # Flip it and convert to tuple for uniqueness check
                flipped_state = tuple(np.flip(state))
                unique_states.add(flipped_state)

        # Convert back to list of numpy arrays
        return [np.array(s) for s in unique_states]

    def decode(self, Y_eff):
        """
        Vectorized Viterbi for the hidden variables.
        Pulls all parameters and data from the parent registry.
        """
        T_eff = len(Y_eff)
        N_joint = self.n_states

        # 1. Emission Probabilities (Log Space)
        # Pulls from self.parents[name]['Z'] and self.parents[name]['phi'] internally
        E = self._emission_prob_all_from_Z_nodes(Y_eff)
        log_E = np.log2(E + 1e-300)

        # 2. Transition Probabilities (Log Space)
        if self.A_joint is None:
            self.A_joint = self._calculate_joint_transition_matrix()
        log_A = np.log2(self.A_joint + 1e-300)

        # 3. Initial Probabilities (Log Space)
        log_pi = np.zeros(N_joint)
        for i in range(N_joint):
            s_tuple = self.idx_to_s_tuple[i]
            log_prob = 0.0
            for m, name in enumerate(self.hidden_parent_names):
                # Add logs instead of multiplying probabilities
                log_prob += np.log2(self.parents[name]["pi"][s_tuple[m]] + 1e-300)
            log_pi[i] = log_prob

        # Normalize log_pi so they sum to 1 in probability space (Log-Sum-Exp trick)
        # Note: logsumexp is usually base e, so for base 2:
        def logsumexp2(x):
            c = np.max(x)
            return c + np.log2(np.sum(2 ** (x - c)))

        log_pi -= logsumexp2(log_pi)

        # 4. Viterbi Recursion
        delta = np.zeros((T_eff, N_joint))
        psi = np.zeros((T_eff, N_joint), dtype=int)
        delta[0] = log_pi + log_E[0]

        for t in range(1, T_eff):
            # Column-wise max: for each current state, find the best previous state
            matrix = delta[t - 1][:, np.newaxis] + log_A
            psi[t] = np.argmax(matrix, axis=0)
            delta[t] = np.max(matrix, axis=0) + log_E[t] #delta[t] = matrix[psi[t], np.arange(N_joint)] + log_E[t]

        # 5. Backtracking
        path = np.zeros(T_eff, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T_eff - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path

    # def predict(self, Y_eff):
    #     T_eff = len(Y_eff)
    #     path = self.decode(Y_eff)
    #
    #     # 1. Base Dynamics: Intercept + AR
    #     Y_hat = self.base_phi[0] + self.base_phi[1] * self.Y_prev
    #
    #     # 2. Add Parent Contributions
    #     for name, data in self.parents.items():
    #         phi = data["phi"]  # Could be length 2 (poly) or n_centers (rbf)
    #
    #         if data.get("is_fixed"):
    #             # Z_fixed_path is (T_eff, K)
    #             Z_fixed = data["Z_fixed_path"]
    #             Y_hat += Z_fixed @ phi
    #         else:
    #             # Hidden parent: Find its index in the joint tuple
    #             m_idx = self.hidden_parent_names.index(name)
    #             Z_dict = data["Z"]
    #
    #             # Efficiently extract the Z values for the Viterbi path
    #             # path[t] is the joint state index, s_tuple[m_idx] is the specific parent state
    #             for t in range(T_eff):
    #                 s_m = self.idx_to_s_tuple[path[t]][m_idx]
    #                 # Z_dict[s_m] is (T_eff, K), we take row t
    #                 Y_hat[t] += np.dot(Z_dict[s_m][t], phi)
    #
    #     return Y_hat, path

    def decode_parent_delays(self, path): # todo: update
        """
        Returns the binary delay vectors for each HIDDEN parent over time.
        Returns: List of arrays, where each array is (T_eff,) containing delay vectors.
        """
        T_eff = len(path)
        # List of length M (number of hidden parents)
        decoded_dict = {name: np.empty(T_eff, dtype=object) for name in self.parents.keys()}

        for t in range(T_eff):
            s_tuple = self.idx_to_s_tuple[path[t]]
            for m, name in enumerate(self.hidden_parent_names):
                pa_idx = self.parent_names.index(name)
                # Get the actual vector (e.g. [1, 0]) from allowed_states
                state_idx = s_tuple[m]
                decoded_dict[name][t] = self.parents[name]["allowed_states"][state_idx]
            for name in self.fixed_parent_names:
                pa_idx = self.parent_names.index(name)
                state_idx = self.parents[name]['fixed_state_indices'][t]
                decoded_dict[name][t] = self.parents[name]["allowed_states"][state_idx]

        return decoded_dict

    def recover_parent_centric_g_u(self, decoded_parent_states): # TODO: update
        """
        Recovers the g_u function (delay index) for each parent, mapping the
        active state back to the original INPUT time index (u).

        This formulation means g_m[u] = k, where k is the lag (0 to L) at which
        parent X_m[u] influenced the output Y.

        :param decoded_parent_states: List of M arrays of binary delay vectors.
        :returns: List of M arrays, where each array g_m is the decoded lag
                  time series for parent m, sized T_eff + L.
        """
        if not decoded_parent_states:
            return []

        # T_eff = len(decoded_parent_states[0])
        T_eff = len(decoded_parent_states[self.parent_names[0]])
        # M = len(self.parent_names)
        L = self.max_lag

        # Total length of the input series (T_eff + L), same as Y's original length
        T_full = T_eff + L

        # Initialize M g_u arrays, filled with -1 (meaning inactive)
        g_u_dict = {name: np.full(T_full, -1, dtype=int) for name in decoded_parent_states.keys()}

        # --- Iteration over Output Time (t_eff) ---

        # For each parent m:
        for pa in g_u_dict.keys():
            g_m = g_u_dict[pa]  # g_m is the array we are filling (size T_full)

            # For each effective time step t_eff (t = L to T_full-1):
            for t_eff in range(T_eff):
                t_global = t_eff + L  # Global time index of the output Y[t_global]
                delay_vector = decoded_parent_states[pa][t_eff]

                # The index 'delay_vec_idx' in the vector (0 to L) is where the value is 1.
                active_delay_vec_indices = np.where(delay_vector == 1)[0]

                # If the state is active (i.e., [0, 1, 1] means active)
                if len(active_delay_vec_indices) > 0:

                    # For every active lag (where the vector element is 1):
                    for delay_vec_idx in active_delay_vec_indices:

                        # 1. Calculate the actual lag (k):
                        # The vector is indexed from 0 (oldest lag, k=L) to L (newest lag, k=0).
                        lag_k = L - delay_vec_idx
                        # lag_k is the value that was stored in your original g_u (0 to L)

                        # 2. Calculate the input time index (u):
                        # The input X_m[u] is used to predict Y[t_global] at lag k.
                        # t_global = u + k  => u = t_global - k
                        u = t_global - lag_k

                        # Guard against negative indices (though t_global >= L should prevent this)
                        if u < T_full and g_m[u] == -1:
                            # 3. Map the lag value (k) back to the input time index (u)
                            # Overwriting occurs here if X_m[u] influences multiple subsequent Y's,
                            # which is the intended behavior based on your request.
                            g_m[u] = lag_k

        return g_u_dict

    def get_param_count(self):
        """
        Return the total number of free parameters (k) across all parents
        (hidden and fixed) and the observation model.
        """
        k = 0

        # --- 1. Initial State Probabilities (pi) & Transition Matrices (A) ---
        for name, data in self.parents.items():
            N_m = len(data["allowed_states"])

            # Free parameters in pi: N_m - 1
            k += (N_m - 1)

            # Free parameters in A:
            # We need the transition mask for THIS specific parent.
            A_mask = data["mask"]
            total_allowed_transitions = np.sum(A_mask)

            # Free parameters = total non-zero entries - number of rows (row-sum constraints)
            k += (total_allowed_transitions - N_m)
        #
        # # --- 1. Initial State Probabilities (pi) & Transition Matrices (A) ---
        # N = sum([len(data["allowed_states"]) for name, data in self.parents.items()])
        # # k += (N - 1)
        # # Free parameters in A:
        # matrices = [data['mask'] for name, data in self.parents.items()]
        # joint_matrix = reduce(np.kron, matrices)
        # total_allowed_transitions = np.count_nonzero(joint_matrix)
        # k += (total_allowed_transitions - N)

        # --- 2. Observation Model (Phi and Sigma) ---
        # base_phi (2) + 2 coefficients per parent (fixed + hidden)
        n_parents = len(self.parents)
        k_Phi = 2 + self.fct_library_size * len(self.parents)

        # Sigma is the one shared variance
        k += (k_Phi + 1)

        return k

    def prune(self, gamma):
        structural_change = False

        # Dictionary to store what we want to keep before we touch the model
        pruning_plan = {}

        # 1. FIRST PASS: Identify states to keep for each hidden parent
        hidden_names = [n for n, d in self.parents.items() if not d.get("is_fixed")]

        for m, name in enumerate(hidden_names):
            parent = self.parents[name]
            old_allowed = parent["allowed_states"]
            N_m = len(old_allowed)

            # # Calculate usage for this specific parent's states
            # visits = np.zeros(N_m)
            # for i in range(self.n_states):
            #     s_m = self.idx_to_s_tuple[i][m]
            #     # if sum(parent["allowed_states"][s_m]) != 1: continue # we only look at "true" states, not transitional states with none or multiple parents
            #     visits[s_m] += np.sum(gamma[:, i])
            #
            # # Determine which indices meet the threshold
            # kept_indices = [idx for idx, val in enumerate(visits) if val >= self.state_visit_threshold]
            #
            # # Fallback: Ensure at least one state remains
            # if not kept_indices:
            #     kept_indices = [np.argmax(visits)]

            kept_indices = [self.idx_to_s_tuple[idx][m] for idx, p_max in enumerate(np.max(gamma, axis=0)) if
                            sum(self.parents[name]['allowed_states'][self.idx_to_s_tuple[idx][m]]) == 1 and (p_max >= self.pruning_threshold or self.idx_to_s_tuple[idx][m] == 0)]

            if len(kept_indices) == 0: return False

            delays = [self.max_lag - np.where(self.parents[name]['allowed_states'][idx] == 1)[0][0] for idx in kept_indices]

            new_states = self.get_allowed_states(delays)
            kept_indices = []
            for state in new_states:
                kept_indices.append([s_idx for s_idx, s in enumerate(self.parents[name]['allowed_states']) if np.array_equal(s, state)][0])

            # kept_indices = [idx for idx, p_max in enumerate(np.max(gamma, axis=0)) if p_max >= self.pruning_threshold or idx==0] # we want to keep a state if its prob to be used is high at least once, and we always keep the empty state
            if not kept_indices:
                kept_indices = [np.argmax(np.max(gamma, axis=0))]

            if len(kept_indices) < N_m:
                structural_change = True

            pruning_plan[name] = {
                "indices": kept_indices,
                "old_pi": parent["pi"].copy(),
                "old_A": parent["A"].copy(),
                "old_allowed": list(old_allowed)
            }

        if not structural_change:
            return False

        # 2. SECOND PASS: Apply changes and Transfer Parameters
        for name in hidden_names:
            plan = pruning_plan[name]
            indices = plan["indices"]
            parent = self.parents[name]

            # Update allowed states
            parent["allowed_states"] = [plan["old_allowed"][i] for i in indices]

            # --- Transfer Pi ---
            new_pi_raw = plan["old_pi"][indices]
            denom = np.sum(new_pi_raw)
            if denom > 1e-15:
                new_pi_raw /= denom
            else:
                # If the probability is effectively zero, reset to uniform
                new_pi_raw = np.ones_like(new_pi_raw) / len(new_pi_raw)
            parent["pi"] = new_pi_raw #new_pi_raw / (np.sum(new_pi_raw) + 1e-12)

            # --- Transfer A ---
            # np.ix_ creates the meshgrid for slicing rows and columns simultaneously
            new_A_raw = plan["old_A"][np.ix_(indices, indices)]

            # Renormalize rows for the new smaller matrix
            row_sums = new_A_raw.sum(axis=1, keepdims=True)
            parent["A"] = np.divide(new_A_raw, row_sums,
                                    out=np.ones_like(new_A_raw) / len(indices),
                                    where=row_sums != 0)

            tmp_A = self.init_transition_matrix(name)
            parent["mask"] = (tmp_A > 0)

        # 3. REBUILD GLOBAL METADATA (Essential for next iteration)
        self.N_states_m = [len(self.parents[n]["allowed_states"]) for n in hidden_names]
        self.joint_state_indices = list(product(*[range(N) for N in self.N_states_m]))
        self.n_states = len(self.joint_state_indices)
        self.idx_to_s_tuple = {i: self.joint_state_indices[i] for i in range(self.n_states)}

        # 4. REBUILD JOINT TRANSITION MATRIX & RE-SLICE DATA
        self.A_joint = self._calculate_joint_transition_matrix()

        # This ensures the OLS regressors match the new idx_to_s_tuple
        # new_Z_data = self._precalculate_Z_per_node(self.X_dict, self.Y)
        # for name in hidden_names:
        #     self.parents[name]["Z"] = new_Z_data[name]
        self._precalculate_Z_per_node(self.X_dict, self.Y)

        total_before = sum(len(plan["old_allowed"]) for plan in pruning_plan.values())
        total_after = self.n_states
        print(f"Pruning: {total_before} -> {total_after} joint states")

        return True

    def get_param_penalty(self, lamb_reg=1, lamb_A=1):
        # return 0
        print(f'Intercept: {self.base_phi[0]:.2f}, AR: {self.base_phi[1]:.2f}')
        # lamb = self.sparsity_pen
        # Regression parameters
        param_pen = np.linalg.norm(self.base_phi)**2 * lamb_reg
        states_pen = 0
        parent_choice = 0
        pruning_pen = 0
        nlogp_reg = sum([-np.log2(scipy.stats.norm(0, 1).pdf(v)) for v in self.base_phi])
        nlogp_A = 0
        A_pen = 0
        for pa, pa_info in self.parents.items():
            nlogp_reg += sum([-np.log2(scipy.stats.norm(0, 1).pdf(v)) for v in pa_info['phi']])
            p_A = self.get_parent_A(pa)
            if len(p_A) == 1:
                nlogp_A += -np.log2(scipy.stats.dirichlet([1]).pdf([1]))
                continue
            for i in range(len(p_A)):
                if i == 0: nlogp_A += -np.log2(scipy.stats.dirichlet([8, 2]).pdf(p_A[i, 1:]))
                elif i == len(p_A) - 1: nlogp_A += -np.log2(scipy.stats.dirichlet([2, 8]).pdf(p_A[i, :-1]))
                else: nlogp_A += -np.log2(scipy.stats.dirichlet([1, 8, 1]).pdf(p_A[i]))

            fcts = [f'func_{n}' for n in range(self.fct_library_size)]
            print(f'Parent {pa}: ', end='')
            for i in range(len(fcts)):
                print(f'{fcts[i]}: {pa_info["phi"][i]:.3f}', end=', ')
            print("")
            reg = np.linalg.norm(pa_info['phi'])**2 * lamb_reg
            # print(f'Penalty for regression params: {reg}')
            # a = np.linalg.norm(pa_info['A'])**2 * self.pruning_threshold #lamb
            # pi = np.linalg.norm(pa_info['pi'])**2 * self.pruning_threshold #lamb
            param_pen += reg
            # param_pen += a
            # param_pen += pi
            # print(f'lambda: {self.sparsity_pen}')
            # print(f'Penalty for regression params: {reg}, A: {a}, pi: {pi}')
            # from scipy.stats import entropy
            # print(f'Shannon entropy of A: {entropy(pa_info['A'], axis=None)}, of pi: {entropy(pa_info['pi'], axis=None)}')
            # refilled_A = np.copy(pa_info['A'])
            # refilled_A.resize((28, 28), refcheck=False)
            # refilled_pi = np.copy(pa_info['pi'])
            # refilled_pi.resize((28, 28), refcheck=False)
            # print(f'Complexity of A: {self.compute_dirichlet_complexity(refilled_A)/28}, of pi: {self.compute_dirichlet_complexity(refilled_pi)/28}')
            # lam_geom = 2
            # N = np.sum(pa_info['A'] > 0)
            # print(f'Geometric probability of the number of valid transitions: {-np.log2(np.exp(-lam_geom*(N**2)))}')
            # print(f'Geometric probability of the number of valid states: {-np.log2(np.exp(-lam_geom*(len(pa_info['allowed_states'])**2)))}') # inclus implicitement les transitions via les etats transitionnels
            # # states_pen -= np.log2(np.exp(-lam_geom*(len(pa_info['allowed_states'])**2)))
            # parent_state = [s for s in pa_info['allowed_states'] if sum(s) == 1]
            # print(f'Geometric probability of the number of valid singleton states: {-np.log2(np.exp(-lam_geom*len(parent_state)))}')
            # states_pen -= np.log2(np.exp(-lam_geom*len(parent_state)))
            # print(f'Combinatorial version: {-np.log2(1/math.comb(self.max_lag+1, len(parent_state)))}')
            # parent_choice -= np.log2(1/math.comb(self.max_lag+1, len(parent_state)))
            # print(f'Parent choice: {-np.log2(1/math.comb(self.max_lag+1, len(parent_state)))}') # bad: penalty based on user defined parameter max_lag
            # pruning_pen -= (np.log2((1-(1-(1-self.pruning_threshold)**(pa_info['initial_n_state']-1))**len(self.Y_eff))) * len(pa_info['allowed_states']))
            # print(f'Pruning penalty: {pruning_pen}')
            # print(f'A complexity penalty: {-(self.compute_dirichlet_marginal_penalty(pa) * lamb_A)}')
            # A_pen -= (self.compute_dirichlet_marginal_penalty(pa) * lamb_A)
            # print(f'A complexity penalty: {self.compute_A_log_prior(pa) * lamb_A}')
            A_pen += (self.compute_A_log_prior(pa) * lamb_A)
            # print(f'pi complexity penalty: {self.compute_pi_prior(pa) * lamb_A}')
            A_pen += (self.compute_pi_prior(pa) * lamb_A)

        # print(f'Penalty for all regression params: {param_pen}') # np.linalg.norm(np.concat([self.base_phi]+[pa_info['phi'] for pa_info in self.parents.values()]))**2 * lamb_reg
        # print(f'Structural cost: ') # prob of having n parents, plus prob of this specific parent combination (does nothing for n_var = 2)

        return nlogp_A + nlogp_reg
        # return param_pen + A_pen #pruning_pen #states_pen #param_pen + states_pen + parent_choice

    def classical_mdl_model_cost(self):
        # A_cost = np.sum([universal_real_encoding(z, 2) for pa_dict in self.parents.values() for z in pa_dict['A'][:, :-1].flatten()])
        A_cost = 0
        for name in self.parents.keys():
            p_A = self.get_parent_A(name)
            A_cost += np.sum([universal_real_encoding(z, 2) for z in p_A[:, [0, 2]].flatten()])
        pi_cost = np.sum([universal_real_encoding(z, 2) for pa_dict in self.parents.values() for z in pa_dict['pi'][:-1]])
        reg_cost = np.sum([universal_real_encoding(z, 2) for pa_dict in self.parents.values() for z in pa_dict['phi']])
        reg_cost += np.sum([universal_real_encoding(z, 2) for z in self.base_phi])

        return A_cost + pi_cost + reg_cost

    def compute_pi_prior(self, name):
        return 0
        pi = self.parents[name]['pi']
        pi_unif = np.ones(shape=(len(pi))) / len(pi)

        kl = np.sum(pi * np.log2((pi + 1e-12) / (pi_unif + 1e-12)))  # KL
        return kl * len(pi)

    def compute_A_log_prior(self, name, kappa=1.0):
        A = self.parents[name]['A']
        A_dummy = self.init_transition_matrix(name)

        # kl = np.sum(A en(A) # scaling by the number of parents

        alpha = kappa * A_dummy + 1e-9
        alpha_sum = np.sum(alpha, axis=1)

        term1 = np.log2(gamma(alpha_sum)) - np.sum(np.log2(gamma(alpha)), axis=1) # term1 = gammaln(alpha_sum) - np.sum(gammaln(alpha), axis=1)
        term2 = np.sum((alpha - 1) * np.log2(A + 1e-12), axis=1)

        log_prior = np.sum(term1 + term2)

        return log_prior


    def compute_dirichlet_marginal_penalty(self, name, kappa=1.0):
        """
        Computes the integrated Dirichlet-Multinomial log-likelihood.

        A_counts: (n_states, n_states) - The expected transition counts from Xi (summed over time).
        A_dummy:  (n_states, n_states) - The uninformative transition probabilities (prior).
        kappa:    Strength of the prior (equivalent to 'virtual' total observations per row).
        """
        # 1. Define Alpha from the uninformative prior
        # Small epsilon to avoid log2(0) in gamma if A_dummy has structural zeros
        A_dummy = self.init_transition_matrix(name)
        A_counts = self.parents[name]['A_counts']

        alpha = kappa * A_dummy + 1e-9

        # 2. Row-wise counts
        # n_ij is A_counts
        # N_i is the total transitions out of state i
        N_i = np.sum(A_counts, axis=1)
        alpha_sum = np.sum(alpha, axis=1)

        # 3. Log Marginal Likelihood calculation (Row by Row)
        # Term 1: ln Gamma(sum alpha) - ln Gamma(N + sum alpha)
        term1 = np.log2(gamma(alpha_sum)) - np.log2(gamma(N_i + alpha_sum)) # term1 = gammaln(alpha_sum) - gammaln(N_i + alpha_sum)

        # Term 2: sum [ ln Gamma(n_ij + alpha_ij) - ln Gamma(alpha_ij) ]
        # Using element-wise operations
        term2 = np.sum(np.log2(gamma(A_counts + alpha)) - np.log2(gamma(alpha)), axis=1) # term2 = np.sum(gammaln(A_counts + alpha) - gammaln(alpha), axis=1)

        # Total Log-Probability for the transitions
        log_marginal_likelihood = np.sum(term1 + term2)

        # log_marginal_likelihood *= len(self.parents[name]['allowed_states']) #*= scale #len(self.Y_eff) # normalization per the number of samples
        # log_marginal_likelihood *= scale

        return log_marginal_likelihood

    def get_nonlinear_library(self, x):
        mode = self.regression
        if mode == 'rbf':
            n_components = 5
            if x is None: return np.zeros(shape=(1, n_components+1)) # to get the size
            rbf_sampler = RBFSampler(gamma=1, n_components=n_components, random_state=42)
            x_2d = np.array(x).reshape(-1, 1)

            rbf_sampler.gamma = 1 / (2 * np.var(x_2d))

            # 1. Get the RBF "bumps"
            z_rbf = rbf_sampler.fit_transform(x_2d)

            # 2. Stack the raw linear x NEXT to the RBFs
            return np.column_stack([x_2d, z_rbf])

        elif mode == 'fct_lib':
            if x is None: x = 1
            return np.column_stack([
                x, x**2, x**3,     # Polynomials
                np.sin(x),         # Periodic
                np.tanh(x),        # Saturation/Sigmoid
                np.exp(-np.abs(x)) # Decay
            ])

        elif mode == 'splines':
            degree = 3
            n_knots = 5
            if x is None: return np.zeros(shape=(1, n_knots - 2)) # to get the size #+ degree id extrapolation not periodic
            # 4 knots creates a very stable, local library
            spline = SplineTransformer(n_knots=n_knots, degree=degree, extrapolation='periodic', include_bias=False, knots='quantile') # include_bias=False to not have the intercept
            x_2d = np.array(x).reshape(-1, 1)
            Z_spline = spline.fit_transform(x_2d)
            return Z_spline

        elif mode == 'poly':
            degree = 5
            if x is None: x = 1
            return np.column_stack([
                x**d for d in range(1, degree+1)     # Polynomials
            ])

    def get_parent_A(self, name):
        A = self.parents[name]['A_counts']
        N = len(self.parents[name]['allowed_states'])
        p_A = np.zeros(shape=(self.max_lag+1, self.max_lag+1))
        for s_idx, vec in enumerate(self.parents[name]['allowed_states']):
            # if sum(vec) != 1: continue # not a parent state
            if sum(vec) == 0: continue # empty state
            pa_lag = self.max_lag - np.where(vec == 1)[0][-1]
            for i in range(N):
                vec2 = self.parents[name]['allowed_states'][i]
                p = A[s_idx, i]
                if p == 0: continue
                if sum(vec2) == 0:
                    # if pa_lag == self.max_lag: continue
                    p_A[pa_lag, pa_lag+1] += p
                else:
                    new_lag = self.max_lag - np.where(vec2 == 1)[0][-1]
                    new_lags = self.max_lag - np.where(vec2 == 1)[0]
                    if new_lag == pa_lag:
                        p_A[pa_lag, new_lag] += p
                    else:
                        seq = [pa_lag] + list(new_lags)
                        for i in range(len(seq)-1):
                            p_A[seq[i], seq[i+1]] += p
        keep = []
        for i in range(self.max_lag+1):
            if np.any(p_A[i, :] != 0) or np.any(p_A[:, i] != 0):
                keep.append(i)
        p_A = p_A[keep][:, keep] # remove unused lags
        p_A = p_A / np.sum(p_A, axis=1).reshape(-1, 1) # normalize
        p_A = np.nan_to_num(p_A)
        for i in range(len(p_A)):
            if sum(p_A[i]) == 0: p_A[i, i] = 1 # to avoid empty rows
        mat = np.zeros(shape=(len(p_A), 3))
        if len(p_A) == 1:
            return np.array([[0, 1, 0]])
        for i in range(len(p_A)):
            if i == 0: mat[i] = [0, p_A[i][i+0], p_A[i][i+1]]
            elif i == len(p_A) - 1: mat[i] = [p_A[i][i-1], p_A[i][i], 0]
            else: mat[i] = p_A[i][i-1:i+2]
        return mat # we would only need to compute the cost of the prob of going up vs going down for each delay

