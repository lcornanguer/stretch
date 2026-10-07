import numpy as np
from numpy.linalg import inv
from itertools import product
import warnings

# Suppress minor division warnings that can occur during initialization/low-probability steps
warnings.filterwarnings("ignore", category=RuntimeWarning)


class FHMM_delay_approx: # TODO: update!
    """
    A Factorial Hidden Markov Model (FHMM) where the hidden state S_t represents a tuple of
    M delays (tau_1, tau_2, ..., tau_M) for M different covariates (X_1, X_2, ..., X_M).

    KEY DIFFERENCE: Transition dynamics for each parent's delay chain are independent,
    but the emission (regression) is shared/coupled.

    The model for hidden state S_t = (tau_1, ..., tau_M) is:
    Y_t = mu + phi_Y * Y_{t-1} +
          SUM_{m=1}^{M} (phi_m1 * X_{m, t-tau_m} + phi_m2 * X_{m, t-tau_m}^2) +
          epsilon_t
    """

    def __init__(self, parent_delays, max_iter=100, tol=1e-3, transition_prior=1.0, verbose=False):
        """
        Initializes the FHMM for M covariates.

        :param parent_delays: A list of lists, where each inner list contains the possible
                              delay values for one parent. E.g., [[1, 2], [3, 4, 5]].
        """
        self.parent_delays = parent_delays
        self.M = len(parent_delays)  # Number of covariates

        if not self.parent_delays:
            raise ValueError("parent_delays cannot be empty.")

        # N_m: Number of states (possible delays) for each parent chain m
        self.N_m = [len(delays) for delays in parent_delays]

        # All possible joint delay combinations (the full state space)
        self.delay_combinations = list(product(*parent_delays))
        self.n_states = len(self.delay_combinations)

        # Mapping: Joint state index (i) -> (tau_1, ..., tau_M) tuple
        self.idx_to_delay = {i: self.delay_combinations[i] for i in range(self.n_states)}

        # Mapping: Parent m's delay (tau_m) -> its index in the small chain (s_m)
        self.delay_to_s_m = [
            {tau: s for s, tau in enumerate(delays)} for delays in parent_delays
        ]

        # Calculate maximum lag needed across all combinations and covariates
        self.max_tau = max(max(delays) for delays in parent_delays) if parent_delays and any(parent_delays) else 0

        # Phi size: 1 (mu) + 1 (phi_Y) + 3*M (for X terms: linear and quadratic per parent)
        self.phi_size = 3 * self.M + 2

        self.max_iter = max_iter
        self.tol = tol
        self.transition_prior = transition_prior
        self.verbose = verbose

        # Parameters initialized to None
        self.A_m = None  # List of independent Transition Matrices [A1, A2, ...]
        self.pi_m = None  # List of independent Initial State Probabilities [pi1, pi2, ...]
        self.Phi = None  # Regression coefficients (phi_size x 1)
        self.Sigma = None  # Error variance (1,)

        # Pre-calculated joint transition matrix A_joint (n_states x n_states)
        self.A_joint = None

        # Create masks to enforce transition constraints
        self.A_permanent_masks = None
        self.A_joint_mask = None

        if self.verbose:
            print(f"Child-centric FHMM Initialized for M={self.M} chains.")
            print(f"Total joint states (n_states): {self.n_states}")

    def _initialize_parameters(self, Y, X_list, warm_init=None):
        """Initializes or loads model parameters."""

        # 1. Initialize M Independent Transition Matrices (A_m)
        self.A_m = []
        self.pi_m = []
        for N in self.N_m:
            # not sure about that init...
            # A_m_init = np.random.rand(N, N)
            # A_m_init[np.diag_indices(N)] += self.transition_prior
            A_m_init = np.full((N, N), self.transition_prior) # uniform probs
            for s1 in range(N):
                for s2 in range(s1+1, N):
                    A_m_init[s1, s2] = 0 # delay must increase one by one

            A_m_init = A_m_init / np.sum(A_m_init, axis=1, keepdims=True)
            self.A_m.append(A_m_init)

            self.pi_m.append(np.ones(N) / N)

        # 2. Shared Regression Coefficients (Phi) and Variance (Sigma) via OLS Warm-up
        if self.Phi is None:

            T = len(Y)
            T_eff = T - self.max_tau

            # Use the state where all parents are at their max lag (most available data)
            max_lag_tuple = tuple(max(d) for d in self.parent_delays)
            # Find the index of this state combination
            try:
                state_idx_max_lag = self.delay_combinations.index(max_lag_tuple)
            except ValueError:
                state_idx_max_lag = 0  # Fallback

            X_OLS, Y_OLS = [], []

            for t_eff in range(T_eff):
                t = t_eff + self.max_tau
                regressors = self._get_regressors(X_list, Y, t, state_idx_max_lag)

                if regressors is not None:
                    X_OLS.append(regressors)
                    Y_OLS.append(Y[t])

            X_OLS = np.array(X_OLS)
            Y_OLS = np.array(Y_OLS)

            # Check for enough data points for OLS
            if X_OLS.shape[0] > self.phi_size:
                # OLS solution: Phi = inv(X^T X) X^T Y
                Phi_new, residuals, rank, s = np.linalg.lstsq(X_OLS, Y_OLS, rcond=None)
                self.Phi = Phi_new

                # Calculate variance based on OLS residuals
                T_OLS = X_OLS.shape[0]
                # residuals is SSE if not zero, otherwise lstsq returns (N,) array of residuals
                SSE = np.sum((Y_OLS - X_OLS @ self.Phi) ** 2)

                # Degrees of freedom: T_OLS - phi_size (number of parameters)
                df = T_OLS - self.phi_size
                if df > 0:
                    self.Sigma = np.array([max(SSE / df, 1e-9)])
                else:
                    self.Sigma = np.array([np.var(Y) / 2.0])
            else:
                # Fallback to random if OLS is not possible
                self.Phi = np.random.rand(self.phi_size) * 0.5 - 0.25
                self.Sigma = np.array([np.var(Y) / 2.0])

        # Pre-calculate A_joint after A_m is initialized
        self.A_joint = self._calculate_joint_transition_matrix()

        # Create masks to enforce transition constraints
        self.A_permanent_masks = [(A > 0) for A in self.A_m]
        self.A_joint_mask = (self.A_joint > 0)

    def _calculate_joint_transition_matrix(self):
        """
        OPTIMIZATION: Pre-calculates the full n_states x n_states joint transition matrix A_joint.
        This is necessary because A_m changes in the M-step.
        """
        A_joint = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):  # previous state (row index)
            for j in range(self.n_states):  # current state (column index)
                # Compute P(S_t=j | S_{t-1}=i)
                A_joint[i, j] = self._joint_transition_prob_internal(i, j)

        return A_joint

    def _get_regressors(self, X_list, Y, t, state_idx):
        """
        Constructs the state-dependent regressor vector Z_t(tau1, ..., tauM).
        Size is 2M + 2.
        """
        # Map joint index to delay tuple
        taus = self.idx_to_delay[state_idx]

        if t < self.max_tau: return None

        # 1. Start with the fixed terms [1.0 (Intercept), Y_{t-1} (AR Term)]
        Y_lag_1 = Y[t - 1]
        regressors = [1.0, Y_lag_1]

        # 2. Loop through all M covariates to add X_m and X_m^2 terms
        for m in range(self.M):
            tau_m = taus[m]
            X_m = X_list[m]

            # Check bounds for the current covariate's lag
            if t - tau_m < 0: return None

            X_lag_tau = X_m[t - tau_m]

            # Add linear and quadratic terms
            regressors.append(X_lag_tau)
            regressors.append(X_lag_tau ** 2)
            regressors.append(X_lag_tau ** 3)

        return np.array(regressors)  # Size 3M + 2

    def _emission_prob(self, X_list, Y, t, state_idx):
        """
        Calculates the probability density of observation Y_t given state S_t.
        """
        regressors = self._get_regressors(X_list, Y, t, state_idx)
        if regressors is None:
            return 1e-10

        # Mean calculation: mu = Phi . Z_t
        mu = np.dot(regressors, self.Phi)

        # Variance (single value) and Residual
        sigma2 = self.Sigma[0]
        residual = Y[t] - mu

        # Gaussian PDF
        if sigma2 <= 1e-9: sigma2 = 1e-9

        log_prob = -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (residual ** 2 / sigma2)
        return np.exp(log_prob)

    def _joint_transition_prob_internal(self, i, j):
        """
        Calculates P(S_t=j | S_{t-1}=i) by multiplying the independent parent transitions.
        This is an internal helper for calculating A_joint.
        """
        taus_i = self.idx_to_delay[i]  # Delay tuple for state i
        taus_j = self.idx_to_delay[j]  # Delay tuple for state j

        joint_prob = 1.0

        for m in range(self.M):
            # 1. Get the small chain index s_m for parent m's delay
            tau_m_i = taus_i[m]
            tau_m_j = taus_j[m]
            s_m_i = self.delay_to_s_m[m][tau_m_i]
            s_m_j = self.delay_to_s_m[m][tau_m_j]

            # 2. Look up the transition probability in the small matrix A_m
            A_m = self.A_m[m]
            transition_prob = A_m[s_m_i, s_m_j]

            # 3. Multiply the independent probabilities
            joint_prob *= transition_prob

        return joint_prob

    def _forward_backward(self, X_list, Y, T_eff):
        """The E-step: Computes the forward (alpha), backward (beta), and smoothed state probabilities (gamma, xi)."""
        # ... (unchanged) ...
        # [Implementation of _forward_backward]
        # Calculate joint initial state probability pi_joint
        pi_temp = np.zeros(self.n_states)
        for i in range(self.n_states):
            taus_i = self.idx_to_delay[i]
            prob = 1.0
            for m in range(self.M):
                tau_m_i = taus_i[m]
                s_m_i = self.delay_to_s_m[m][tau_m_i]
                prob *= self.pi_m[m][s_m_i]
            pi_temp[i] = prob
        pi_joint = pi_temp / np.sum(pi_temp)  # Normalize in case of small numerical errors

        # --- PRE-CALCULATE EMISSION PROBABILITIES (E_t) ---
        E = np.zeros((T_eff, self.n_states))
        for t in range(T_eff):
            E[t, :] = np.array([self._emission_prob(X_list, Y, t + self.max_tau, i) for i in range(self.n_states)])

        # Alpha (Forward Pass)
        alpha = np.zeros((T_eff, self.n_states))

        # Initialization
        alpha[0, :] = pi_joint * E[0, :]

        scale = np.sum(alpha[0, :])
        alpha[0, :] /= scale
        log_likelihood = np.log(scale)

        # Recursion (VECTORIZED: O(T * N_joint^2) but using fast matrix multiplication)
        for t in range(1, T_eff):
            prediction = alpha[t - 1, :] @ self.A_joint
            alpha[t, :] = E[t, :] * prediction

            scale = np.sum(alpha[t, :])
            if scale == 0:
                alpha[t, :] = alpha[t - 1, :]
            else:
                alpha[t, :] /= scale
                log_likelihood += np.log(scale)

        # Beta (Backward Pass)
        beta = np.zeros((T_eff, self.n_states))
        beta[T_eff - 1, :] = 1.0

        # Recursion (VECTORIZED)
        for t in range(T_eff - 2, -1, -1):
            beta[t, :] = self.A_joint @ (E[t + 1, :] * beta[t + 1, :])
            beta[t, :] /= np.sum(beta[t, :])

        # Gamma (Smoothed State Probabilities)
        gamma = alpha * beta
        gamma = gamma / np.sum(gamma, axis=1, keepdims=True)

        # Xi (Joint Smoothed Probabilities)
        xi = np.zeros((T_eff - 1, self.n_states, self.n_states))
        for t in range(T_eff - 1):
            E_beta_term = E[t + 1, :] * beta[t + 1, :]
            numerator = alpha[t, :].reshape(-1, 1) * self.A_joint * E_beta_term.reshape(1, -1)
            denom = np.sum(numerator)
            if denom > 1e-10:
                xi[t, :, :] = numerator / denom
            else:
                xi[t, :, :] = np.ones((self.n_states, self.n_states)) / (self.n_states ** 2)

        return log_likelihood, gamma, xi

    def _maximization_step(self, X_list, Y, T_eff, gamma, xi):
        """The M-step: Re-estimates A_m, pi_m, Phi (Size 2M+2), and Sigma."""
        # ... (unchanged) ...
        # [Implementation of _maximization_step]
        # --- 1. Re-estimate Initial State Probabilities (pi_m) ---
        for m in range(self.M):
            N_m = self.N_m[m]
            pi_m_new = np.zeros(N_m)
            for i in range(self.n_states):
                taus_i = self.idx_to_delay[i]
                tau_m = taus_i[m]
                s_m = self.delay_to_s_m[m][tau_m]
                pi_m_new[s_m] += gamma[0, i]
            self.pi_m[m] = pi_m_new / np.sum(pi_m_new)

        # --- 2. Re-estimate Independent Transition Matrices (A_m) ---
        for m in range(self.M):
            N_m = self.N_m[m]
            expected_transitions_m = np.zeros((N_m, N_m))
            A_mask = self.A_permanent_masks[m]
            for i in range(self.n_states):
                for j in range(self.n_states):
                    taus_i = self.idx_to_delay[i]
                    taus_j = self.idx_to_delay[j]
                    s_m_i = self.delay_to_s_m[m][taus_i[m]]
                    s_m_j = self.delay_to_s_m[m][taus_j[m]]
                    expected_transitions_m[s_m_i, s_m_j] += np.sum(xi[:, i, j])

            # prior_matrix = np.eye(N_m) * self.transition_prior # i don't like that
            # regularized_transitions = expected_transitions_m + prior_matrix
            regularized_transitions = (expected_transitions_m + self.transition_prior) * A_mask
            expected_visits = np.sum(regularized_transitions, axis=1, keepdims=True)
            expected_visits[expected_visits == 0] = 1
            self.A_m[m] = regularized_transitions / expected_visits

        self.A_joint = self._calculate_joint_transition_matrix()

        # --- 3. Re-estimate SHARED Regression Coefficients (Phi) and Variances (Sigma) ---
        Y_eff = Y[self.max_tau:]
        T_eff = len(Y_eff)
        Total_Covariance = np.zeros((self.phi_size, self.phi_size))
        Total_Cross_Covariance = np.zeros(self.phi_size)
        Total_Weighted_SSE = 0.0

        for t in range(T_eff):
            Y_t = Y_eff[t]
            for state_idx in range(self.n_states):
                gamma_t_i = gamma[t, state_idx]
                regressors = self._get_regressors(X_list, Y, t + self.max_tau, state_idx)

                if regressors is not None and gamma_t_i > 1e-10:
                    Z_t_i_T = regressors.reshape(-1, 1)
                    Z_t_i = regressors.reshape(1, -1)
                    Total_Covariance += gamma_t_i * (Z_t_i_T @ Z_t_i)
                    Total_Cross_Covariance += gamma_t_i * regressors * Y_t

        if np.linalg.det(Total_Covariance) > 1e-9:
            Phi_new = inv(Total_Covariance) @ Total_Cross_Covariance
            self.Phi = Phi_new
        else:
            if self.verbose: print("Warning: Pooled WLS matrix singular. Keeping old Phi.")

        for t in range(T_eff):
            Y_t = Y_eff[t]
            for state_idx in range(self.n_states):
                gamma_t_i = gamma[t, state_idx]
                regressors = self._get_regressors(X_list, Y, t + self.max_tau, state_idx)

                if regressors is not None and gamma_t_i > 1e-10:
                    prediction = np.dot(regressors, self.Phi)
                    residual = Y_t - prediction
                    Total_Weighted_SSE += gamma_t_i * residual ** 2

        expected_visits_total = T_eff
        sigma2_new = Total_Weighted_SSE / expected_visits_total
        self.Sigma[0] = max(sigma2_new, 1e-9)

    def fit(self, X_list, Y, warm_init=None):
        """
        Performs the EM algorithm to estimate FHMM parameters.
        """
        T = len(Y)
        T_eff = T - self.max_tau

        if len(X_list) != self.M:
            raise ValueError(f"Input X_list has {len(X_list)} covariates, but model expected {self.M}.")
        if T_eff <= 0:
            raise ValueError(f"Time series length ({T}) is too short. Need at least {self.max_tau + 1} observations.")

        # Initialize parameters: Pass X_list to allow OLS warm-up
        self._initialize_parameters(Y, X_list, warm_init)

        log_likelihood_history = []

        if self.verbose:
            print(f"Starting EM optimization for FHMM with {self.M} independent chains...")

        for iteration in range(self.max_iter):
            # E-Step (Uses joint state space, but factorized A matrix)
            log_likelihood, gamma, xi = self._forward_backward(X_list, Y, T_eff)

            # M-Step (Updates independent A_m and pi_m, but coupled Phi and Sigma)
            self._maximization_step(X_list, Y, T_eff, gamma, xi)

            log_likelihood_history.append(log_likelihood)

            # Check for convergence
            if iteration > 0:
                delta_L = log_likelihood - log_likelihood_history[-2]
                if self.verbose and iteration%10 == 0:
                    print(
                        f"Iteration {iteration + 1}/{self.max_iter}: Log-Likelihood = {log_likelihood:.4f}, Delta = {delta_L:.6f}")
                if np.abs(delta_L) < self.tol:
                    if self.verbose: print(f"Convergence reached at iteration {iteration + 1}. Log-Likelihood = {log_likelihood_history[-1]}")
                    break
                if not np.isfinite(log_likelihood):
                    if self.verbose:
                        print(f"NaN or Inf log-likelihood at iteration {iteration + 1}, stopping.")
                    break
        else:
            if self.verbose: print(f"Maximum iterations ({self.max_iter}) reached without convergence.")

        return log_likelihood_history[-1], gamma

    def decode_states(self, X_list, Y):
        """Viterbi Algorithm: Finds the single most likely sequence of hidden states (delay tuples)."""

        # --- ADDED SAFETY CHECK ---
        if self.pi_m is None or self.A_m is None or self.Phi is None or self.Sigma is None:
            raise ValueError(
                "Model parameters are not initialized. You must call model.fit(X_list, Y) successfully before calling decode_states()."
            )
        # --- END ADDED SAFETY CHECK ---

        # Calculate joint initial state probability pi_joint
        pi_temp = np.zeros(self.n_states)
        for i in range(self.n_states):
            taus_i = self.idx_to_delay[i]
            prob = 1.0
            for m in range(self.M):
                tau_m_i = taus_i[m]
                s_m_i = self.delay_to_s_m[m][tau_m_i]
                prob *= self.pi_m[m][s_m_i]
            pi_temp[i] = prob
        pi_joint = pi_temp / np.sum(pi_temp)

        T = len(Y)
        T_eff = T - self.max_tau

        delta = np.zeros((T_eff, self.n_states))
        psi = np.zeros((T_eff, self.n_states), dtype=int)

        # Ensure A_joint is current
        if self.A_joint is None:
            self.A_joint = self._calculate_joint_transition_matrix()

        # Pre-calculate log emissions
        log_E = np.zeros((T_eff, self.n_states))
        for t in range(T_eff):
            log_E[t, :] = np.log([self._emission_prob(X_list, Y, t + self.max_tau, i) for i in range(self.n_states)])

        # Pre-calculate log transitions (Log A_joint)
        log_A_joint = np.log(self.A_joint)

        # Initialization
        for i in range(self.n_states):
            # Only use the effective time steps starting at self.max_tau
            delta[0, i] = np.log(pi_joint[i]) + log_E[0, i]

        # Recursion (VECTORIZED for inner loop)
        for t in range(1, T_eff):
            # transition_scores = delta[t - 1, i] + log_A_joint[i, j]
            # Max_score = max_i(transition_scores) + log_E[t, j]

            # Sum of transition scores from ALL previous states i to current state j
            # delta[t-1, :].reshape(-1, 1) adds the scores down the columns
            transition_scores = delta[t - 1, :].reshape(-1, 1) + log_A_joint

            # Max over previous states (axis=0) for each current state j
            max_scores = np.max(transition_scores, axis=0)

            delta[t, :] = max_scores + log_E[t, :]
            psi[t, :] = np.argmax(transition_scores, axis=0)  # Index of previous state i

        # Termination: Find the best last state
        last_state_idx = np.argmax(delta[T_eff - 1, :])

        # Path Backtracking
        path = np.zeros(T_eff, dtype=int)
        path[T_eff - 1] = last_state_idx
        for t in range(T_eff - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        # Map state indices to actual delay pairs
        decoded_delays = np.array([self.idx_to_delay[i] for i in path], dtype=object)

        return decoded_delays

    def predict(self, X_list, Y):
        """
        Reconstructs the time series Y_hat based on the most likely sequence
        of hidden states (Viterbi path) and the learned parameters.
        """
        # 1. Decode the most likely state sequence
        try:
            decoded_delays = self.decode_states(X_list, Y)
        except ValueError as e:
            print(f"Error during prediction: {e}")
            return Y.copy()  # Return copy of original Y if decoding fails

        T = len(Y)
        T_eff = T - self.max_tau

        # Initialize Y_hat as a copy of Y. The first max_tau points are unpredicted.
        Y_hat = Y.copy()

        # Loop through the effective time points (from max_tau up to T-1)
        for t_eff in range(T_eff):
            t = t_eff + self.max_tau

            # The state is the decoded delay tuple
            delay_tuple = decoded_delays[t_eff]

            # Find the index of the delay tuple in the model's combinations
            try:
                # The delay tuple is guaranteed to be in the map if decode_states worked
                state_idx = self.delay_combinations.index(delay_tuple)
            except ValueError:
                continue

            # Get regressors Z_t for the predicted state
            regressors = self._get_regressors(X_list, Y, t, state_idx)

            if regressors is not None:
                # Prediction: mu = Phi . Z_t
                Y_hat[t] = np.dot(regressors, self.Phi)

        return Y_hat


# -------------------------------------------------------------------------------------------------------------------
# --- UTILITY FUNCTIONS FOR BIC SCORE CALCULATION (UNCHANGED) ---

def calculate_effective_states(gamma, threshold=1.0):
    """
    Calculates the effective number of used states (N_eff) based on the
    total expected visits (sum of gamma).
    """
    if gamma is None or gamma.size == 0:
        return 0

    expected_visits = np.sum(gamma, axis=0)
    n_eff = np.sum(expected_visits > threshold)

    return max(1, n_eff)


def get_hmm_n_params(n_eff_states_m, n_covariates):
    """
    Calculates the effective number of free parameters (k_eff) for the
    FHMM_Multivariate_SAR model (sum of independent chain penalties + coupled emission).

    :param n_eff_states_m: List of effective number of states for each parent [N_eff1, N_eff2, ...].
    :param n_covariates: Number of covariates (M).
    :returns: Total effective number of free parameters (k_eff).
    """
    k_transition_eff = 0

    for N_eff in n_eff_states_m:
        if N_eff > 0:
            # pi_m: N_eff - 1
            # A_m: N_eff * (N_eff - 1)
            k_transition_eff += (N_eff - 1) + (N_eff * (N_eff - 1))

    # Fixed Parameters for the Emission Distribution (Phi and Sigma)
    # Phi size is 2*M (X terms) + 2 (mu and phi_Y)
    # Sigma size is 1 (variance)
    k_emission_fixed = (2 * n_covariates + 2) + 1

    return k_transition_eff + k_emission_fixed


def calculate_bic_score(log_likelihood, n_params, t_eff):
    """
    Calculates the Bayesian Information Criterion (BIC) score (MDL approximation).
    BIC = k * log(T_eff) - 2 * log_likelihood

    :param log_likelihood: The Log-Likelihood (L) of the model.
    :param n_params: The number of free parameters (k).
    :param t_eff: The effective number of data points.
    :returns: The BIC score (float). Lower is better.
    """
    if t_eff <= 0 or n_params <= 0:
        return np.inf

    return n_params * np.log(t_eff) - 2 * log_likelihood
