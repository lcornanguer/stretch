import math

import numpy as np
import torch

from .child_centric_em import calculate_bic_score, FHMM_delay_approx
from .utils import universal_real_encoding, universal_integer_encoding
from .parent_centric_em import FHMM_ParentSet


def fit_edge(covariates: list,
               target: int,
                global_params, verbosity, data, model_cache,
               learn_sigma=True,
               eps=1e-8,
                lamb_mdl=1
               ) -> int:
    if verbosity > 0:
        print(f'\tEval edge {covariates}->{target}')

    init_with_child_centric = global_params["init_with_child_centric"]

    T_eff = len(data[target]) - global_params["max_lag"]  # check number of samples

    if len(covariates) == 0:
        # LR
        X = torch.tensor(data[target], dtype=torch.float64)
        X_prev = X[:-1].unsqueeze(1)  # [T-1,1]
        X_target = X[1:].unsqueeze(1)  # [T-1,1]

        # Create the column of ones for intercept
        ones = torch.ones_like(X_prev)

        # Concatenate to create [T-1, 2] matrix: [X_prev, 1]
        X_augmented = torch.cat([X_prev, ones], dim=1)

        # Solve
        lsq_result = torch.linalg.lstsq(X_augmented, X_target)
        weights = lsq_result.solution

        coeff = weights[0]
        intercept = weights[1]

        # Prediction
        X_pred = X_augmented @ weights
        X_pred = X_pred.squeeze(-1)  # shape [T-1], safely
        X_target = X_target.squeeze(-1)  # shape [T-1], safely

        residuals_X = X_target - X_pred

        if learn_sigma:
            # Use MLE variance: sum(sq_resid) / N
            mse = (residuals_X ** 2).mean()
            sigma_X = mse + eps
            # Simplified NLL for Gaussian: (N/2) * log2(2*pi*sigma^2) + N/2
            # The second term is N/2 because (residuals**2).sum() / (2 * residuals.mean()) = N/2
            nll = 0.5 * T_eff * torch.log2(2 * torch.pi * sigma_X) + 0.5 * T_eff / torch.log(torch.tensor(2.0))
            nll = nll.item()
        else:
            sigma_X = torch.tensor(1)
            nll = 0.5 * T_eff * torch.log2(2 * torch.pi * sigma_X) + (residuals_X ** 2).sum() / (2 * sigma_X) / torch.log(torch.tensor(2.0))
            nll = nll.item()

        bic_hmm_eff = calculate_bic_score(-nll, 2, T_eff)
        print(f"BIC: {bic_hmm_eff}")

        lamb = 1 #global_params['lambda']
        param_pen = lamb * torch.norm(weights) ** 2
        david_score = param_pen + nll# sse

        mdl_cost = nll + np.sum([universal_real_encoding(w, 2) for w in weights.flatten().tolist()])
        print(f"MDL: {mdl_cost}")

        # sigma_sq is fixed from your null model
        sse = (residuals_X ** 2).sum()
        data_fit_nats = sse / (2 * sigma_X)
        constant_nats = (T_eff / 2) * np.log2(2 * np.pi * sigma_X)
        # Your transition log_likelihood is already in nats
        transition_nats = 0  # -hmm_log_likelihood
        # david_score = data_fit_nats + constant_nats + param_pen + transition_nats
        # print(f"DS: {david_score}")

        if not global_params['use_bic']:
            bic_hmm_eff = mdl_cost #david_score

        # if verbosity>0:
        score_marg = 0
        print(f'\t{covariates} -> {target}: nll={np.round(nll, 2)}, residuals={sum(residuals_X)}, sigma={sigma_X}')

        # Save score and models
        cached_model = {}
        cached_model['model'] = None
        cached_model['score'] = bic_hmm_eff
        cached_model['instantaneous'] = {}
        model_cache[f'j_{str(target)}_pa_{covariates}'] = cached_model

        return bic_hmm_eff
    else:
        Y = data[target]
        X = [data[d] for d in covariates]
        X_dict = {d: data[d] for d in covariates}

        max_delay = global_params["max_lag"]

        if init_with_child_centric and len(covariates) == 1:
            T = len(Y)
            DELAYS = [i for i in range(max_delay + 1)]
            PARENT_DELAYS = [DELAYS for _ in covariates]
            N_m_list = [len(d) for d in PARENT_DELAYS]
            child_centric_model = FHMM_delay_approx(
                parent_delays=PARENT_DELAYS,  # Use the new input format
                tol=T * 1e-6,  # 1e-6,
                verbose=global_params["verbose_iter"],
                max_iter=250
            )
            child_centric_model.fit(X, Y)
            # --- RESULTS AND DECODING ---
            decoded_delays = child_centric_model.decode_states(X, Y)

            # translate g_u from child-centric decoded_delays
            T = len(Y)
            parent_sets = list()
            for pa in range(decoded_delays.shape[1]):
                g_u = np.full(shape=len(X[pa]),
                              fill_value=-1)  # delay at the parent time, size: T (last max_delay values should be filled with -1 or u)
                ch_index = np.full(shape=len(X[pa]),
                                   fill_value=-1)  # child index that will receive effect from parent u, size: T (last max_delay values should be filled with -1 or u)
                tmp = np.array([t + max(DELAYS) - delay for t, delay in enumerate(decoded_delays[
                                                                                      :, pa])])  # contains temporary parent index for each t (first values skipped, time series index starts at max_delay)
                for u in range(T - 1, -1, -1):  # decoded_delays.shape[0]
                    children = np.where(tmp == u)[0]
                    children += max(DELAYS)
                    if len(children) > 1:
                        ch_index[u] = min(children)
                    elif len(children) == 0:
                        ch_index[u] = ch_index[u + 1] if u + 1 < T else u
                    else:
                        ch_index[u] = children[0]
                    g_u[u] = ch_index[u] - u

                # translate g_u to parent set
                parent_sets_m = np.zeros((len(ch_index), len(DELAYS)))  # parent_set at time t, size: T
                for u, delay in enumerate(g_u):
                    parent_sets_m[u + delay, max(DELAYS) - delay] = 1
                parent_sets.append(parent_sets_m)

        T = len(Y)
        if len(covariates) >= 2:
            if global_params["fix_previous_parents"]:
                parent_centric_model = FHMM_ParentSet(
                    max_lag=global_params["max_lag"],
                    max_iter=250,
                    tol=T * 1e-6,
                    verbose=global_params["verbose_iter"],
                    parent_names=[c for c in covariates],
                    use_lasso=global_params["use_lasso"],
                    regression=global_params["regression"],
                    pruning_threshold=global_params["pruning_threshold"]
                )
                current_parents = covariates[:-1]
                model = model_cache[f'j_{str(target)}_pa_{current_parents}']['model']
                print(f"Previously fitted parents ({current_parents}) fixed in new model.")
                parent_centric_model.add_fixed_parents(model)
            else:
                # raise NotImplementedError()
                parent_centric_model = FHMM_ParentSet(
                    max_lag=global_params["max_lag"],
                    max_iter=250,
                    tol=T * 1e-6,
                    verbose=global_params["verbose_iter"],
                    pruning=True,
                    parent_names=[c for c in covariates],
                    use_lasso=global_params["use_lasso"],
                    regression=global_params["regression"],
                    pruning_threshold=global_params["pruning_threshold"]
                )
                if len(covariates) >= 3: # memory issue beyond
                    current_parents = covariates[:-2]
                    model = model_cache[f'j_{str(target)}_pa_{current_parents}']['model']
                    print(f"Previously fitted parents ({current_parents}) fixed in new model.")
                    parent_centric_model.add_fixed_parents(model)
            final_ll = parent_centric_model.fit(X_dict, Y)
        else:
            parent_centric_model = FHMM_ParentSet(
                max_lag=global_params["max_lag"],
                max_iter=250,
                tol=T * 1e-6,
                verbose=global_params["verbose_iter"],
                parent_names=[c for c in covariates],
                use_lasso=global_params["use_lasso"],
                regression=global_params["regression"],
                pruning_threshold=global_params["pruning_threshold"]
            )
            if init_with_child_centric:
                print("Warm initialization using the child-centric approximation.")
                final_ll = parent_centric_model.fit(X_dict, Y,
                                                    warm_init={c: parent_sets[i] for i, c in enumerate(covariates)})
            else:
                final_ll = parent_centric_model.fit(X_dict, Y)  # no warm init
        # Y_hat, g_u = parent_centric_model.predict(X, Y)

        if verbosity:
            print(f'\t{covariates} -> {target}: nll={np.round(-final_ll, 2)}')

        # CALCULATE EFFECTIVE PARAMETERS
        n_params = parent_centric_model.get_param_count()
        bic_hmm = calculate_bic_score(final_ll, n_params, T_eff)
        david_score = -final_ll + parent_centric_model.get_param_penalty()

        if 'lamb_mdl' in global_params: lamb_mdl = global_params['lamb_mdl']
        parent_identification_cost = universal_integer_encoding(len(covariates)) + np.log2(math.comb(len(data)-1, len(covariates)))
        mdl_cost = -final_ll + lamb_mdl*(parent_centric_model.classical_mdl_model_cost() + parent_identification_cost)
        print(f"BIC: {bic_hmm}")
        # print(f"DS: {david_score}")
        print(f"MDL: {mdl_cost}")

        if not global_params['use_bic']:
            bic_hmm = mdl_cost #david_score

        path = parent_centric_model.decode(parent_centric_model.Y_eff)
        decoded_dict = parent_centric_model.decode_parent_delays(path)
        g_u = parent_centric_model.recover_parent_centric_g_u(decoded_dict)
        for pa, delays in g_u.items():
            delays = [d for d in delays if d != -1]
            print(
                f"Delay with parent {pa}: avg={np.mean(delays):.2f}, std={np.std(delays):.2f}, min={np.min(delays)}, max={np.max(delays)}")

        # Save score and models
        cached_model = {}
        cached_model['model'] = parent_centric_model
        cached_model['score'] = bic_hmm
        cached_model['instantaneous'] = {pa: sum(np.array(delays) == 0)/len(delays) > 0.2 for pa, delays in g_u.items()} # weak test for acyclic full causal graph, only work for correctly fitted constant delay
        model_cache[f'j_{str(target)}_pa_{covariates}'] = cached_model

        return bic_hmm
