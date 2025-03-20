from typing import Optional, Union, List
import time as time
import numpy as np

class SFFSAValueIteration:

    def __init__(self, 
                 env, 
                 gpi, 
                 constraint:Optional[dict] = None) -> None:
        
        self.env = env 
        self.fsa = self.env.fsa 
        self.gpi = gpi 
        self.exit_states = self.env.exit_states
        self.constraint = constraint

    def traverse(self, 
                 weights: Optional[dict] = None, 
                 num_iters: Optional[dict] = 100) -> Union[dict, List[float]]:

        exit_states = self.exit_states

        U = self.fsa.states
        
        if weights is None:       
            W = np.zeros((len(U), self.env.feat_dim))
            # For each fsa state we have a vector w of size phi that indicates
            # how important each goal is given that we're in that fsa state
        else:
            W = np.asarray(list(weights.values()))

        timemarks = [0]
        
        for _ in range(num_iters):

            start_time = time.time()

            W_old = W.copy()  # Keep to compare diff with new weights for stopping iteration

            # For each possible starting state
            for (uidx, u) in enumerate(U):
                
                if self.fsa.is_terminal(u):
                    continue
                
                weights = []

                for (vidx, v) in enumerate(U):
                    # For each transition
                    # (i.e. for each other state v for which there exists an edge between u and v (transition))
                    if not self.fsa.graph.has_edge(u, v):
                        continue

                    w = np.zeros((self.env.feat_dim))  # We define a task vector of size phi_dim

                    # Get the predicates that satisfy the transition
                    propositions = self.fsa.get_predicate((u, v)) 
                    idxs = [self.fsa.symbols_to_phi[prop] for prop in propositions]  # this gets the indices of the
                    # propositions

                    if self.fsa.is_terminal(v):
                        # If the transition is terminal we get a w of 1 for each proposition that enables this
                        # transition
                        w[idxs] = 1
                    else:
                        for idx in idxs:
                            e = exit_states[idx]  # TODO: This is now a set in my officeAreas case

                            if isinstance(e, set):
                                q_vals = []
                                for exit_state in e:
                                    q_val_e = np.dot(self.gpi.max_q(exit_state, W[vidx]), W[vidx])
                                    q_vals.append(q_val_e)
                                w[idx] = np.mean(q_vals)  # Take the mean as a temporary solution
                            else:
                                w[idx] = np.dot(self.gpi.max_q(e, W[vidx]), W[vidx])  # Assign to each goal/predicate in w
                                # the maximum Q-value we can get in its corresponding exit state (over policies and actions)
                                # 'Given the current task vector w in fsa state v, what is the max Q-value?'

                            # TODO: For sets of exit state, also max over exit states?

                    weights.append(w)
                
                weights = np.asarray(weights)
                weights = np.sum(weights, axis=0)  # Sum the rows, so we get w of size phi_dim where each element
                # is the sum over all transitions from u to v1, v2, v3 etc.
                # TODO: Does this make sense? since if we have two transitions for a single goal/proposition we will sum
                # TODO: the Q-values and then have a higher weight than for goals that only have one transition
                # TODO: while we cannot simultaniously take both transitions so MAX might make more sense?
                # TODO: at the same time if deterministic then when we satisfy one proposition we will enter both
                # TODO: transitions which is not possible so this would only be the case in stoch setting?
                
                if self.constraint:
                    for c in self.constraint:
                        weights[c] = self.constraint[c]

                W[uidx] = weights
            
            elapsed_time = time.time() - start_time
            timemarks.append(elapsed_time)

            if np.allclose(W, W_old):
                break

        W = {u: W[U.index(u)] for u in U}

        return W, np.cumsum(timemarks)


class SFFSAValueIterationRBFCentersOnly:

    def __init__(self,
                 env,
                 gpi,
                 constraint: Optional[dict] = None) -> None:

        self.env = env
        self.fsa = self.env.fsa
        self.gpi = gpi
        self.exit_states = self.env.exit_states
        self.constraint = constraint

    def traverse(self,
                 weights: Optional[dict] = None,
                 num_iters: Optional[dict] = 100) -> Union[dict, List[float]]:

        exit_states_mapping = self.exit_states

        U = self.fsa.states

        if weights is None:
            W = np.zeros((len(U), self.env.feat_dim))
            # For each fsa state we have a vector w of size phi that indicates
            # how important each goal is given that we're in that fsa state
        else:
            W = np.asarray(list(weights.values()))

        timemarks = [0]

        for _ in range(num_iters):

            start_time = time.time()

            W_old = W.copy()  # Keep to compare diff with new weights for stopping iteration

            # For each possible starting state
            for (uidx, u) in enumerate(U):

                if self.fsa.is_terminal(u):
                    continue

                weights = []

                for (vidx, v) in enumerate(U):
                    # For each transition
                    # (i.e. for each other state v for which there exists an edge between u and v (transition))
                    if not self.fsa.graph.has_edge(u, v):
                        continue

                    w = np.zeros((self.env.feat_dim))  # We define a task vector of size phi_dim

                    # Get the predicate that satisfies the transition
                    proposition = self.fsa.get_predicate((u, v))
                    assert len(proposition) == 1, "Only one predicate is supported"
                    proposition = proposition[0]

                    rbf_centers = self.env.env.COORDS_RBFS[proposition]
                    rbf_weight_indices = [self.env.env.rbf_indices[center] for center in rbf_centers]

                    if self.fsa.is_terminal(v):
                        # If the transition is terminal we get a w of 1 for each proposition that enables this
                        # transition
                        w[rbf_weight_indices] = 1
                    else:
                        for (idx, center_coords) in zip(rbf_weight_indices, rbf_centers):
                            w[idx] = np.dot(self.gpi.max_q(center_coords, W[vidx]), W[vidx])

                    weights.append(w)

                # Sum the rows, so we get w of size phi_dim where each element
                # is the sum over all transitions from u to v1, v2, v3 etc.
                weights = np.asarray(weights)
                weights = np.sum(weights, axis=0)

                W[uidx] = weights

            elapsed_time = time.time() - start_time
            timemarks.append(elapsed_time)

            if np.allclose(W, W_old):
                break

        W = {u: W[U.index(u)] for u in U}

        return W, np.cumsum(timemarks)


class SFFSAValueIterationMean:

    def __init__(self,
                 env,
                 gpi,
                 constraint: Optional[dict] = None) -> None:

        self.env = env
        self.fsa = self.env.fsa
        self.gpi = gpi
        self.exit_states = self.env.exit_states
        self.constraint = constraint

    def traverse(self,
                 weights: Optional[dict] = None,
                 num_iters: Optional[dict] = 100) -> Union[dict, List[float]]:

        exit_states = self.exit_states

        U = self.fsa.states

        if weights is None:
            W = np.zeros((len(U), self.env.feat_dim))
            # For each fsa state we have a vector w of size phi that indicates
            # how important each goal is given that we're in that fsa state
        else:
            W = np.asarray(list(weights.values()))

        timemarks = [0]

        for _ in range(num_iters):

            start_time = time.time()

            W_old = W.copy()  # Keep to compare diff with new weights for stopping iteration

            # For each possible starting state
            for (uidx, u) in enumerate(U):

                if self.fsa.is_terminal(u):
                    continue

                weights = []

                for (vidx, v) in enumerate(U):
                    # For each transition
                    # (i.e. for each other state v for which there exists an edge between u and v (transition))
                    if not self.fsa.graph.has_edge(u, v):
                        continue

                    w = np.zeros((self.env.feat_dim))  # We define a task vector of size phi_dim

                    # Get the predicate that satisfies the transition
                    proposition = self.fsa.get_predicate((u, v))
                    assert len(proposition) == 1, "Only one predicate is supported"
                    proposition = proposition[0]

                    feature_idxs = np.where(np.array(self.env.env.prop_at_feat_idx) == proposition)[0]
                    exit_states = self.exit_states[self.fsa.symbols_to_phi[proposition]]  # This is a set with all
                    # states that are in the desired Area

                    if self.fsa.is_terminal(v):
                        # If the transition is terminal we get a w of 1 for each proposition that enables this
                        # transition
                        w[feature_idxs] = 1
                    else:
                        q_vals = []
                        feature_weights = {feature_idx: [] for feature_idx in feature_idxs}
                        for exit_state in exit_states:
                            q_val = np.dot(self.gpi.max_q(exit_state, W[vidx]), W[vidx])
                            phi = self.env.env.features(state=None, action=None, next_state=exit_state)
                            q_vals.append(q_val)

                            for feature_idx in feature_idxs:
                                feature_weight = phi[feature_idx]
                                feature_weights[feature_idx].append(feature_weight)

                        q_vals = np.array(q_vals)

                        # Normalize weights for each feature index
                        for feature_idx in feature_idxs:
                            feature_weights_arr = np.array(feature_weights[feature_idx])  # Convert to NumPy array

                            # Normalize weights so they sum to 1
                            sum_feature_weights = np.sum(feature_weights_arr)
                            if sum_feature_weights != 0:
                                normalized_weights = feature_weights_arr / sum_feature_weights
                            else:
                                normalized_weights = np.zeros_like(feature_weights_arr)  # Avoid division by zero

                            # Compute weighted mean
                            w[feature_idx] = np.sum(normalized_weights * q_vals)  # Weighted sum

                    weights.append(w)

                weights = np.asarray(weights)
                weights = np.sum(weights, axis=0)  # Sum the rows, so we get w of size phi_dim where each element
                # is the sum over all transitions from u to v1, v2, v3 etc.

                if self.constraint:
                    for c in self.constraint:
                        weights[c] = self.constraint[c]

                W[uidx] = weights

            elapsed_time = time.time() - start_time
            timemarks.append(elapsed_time)

            if np.allclose(W, W_old):
                break

        W = {u: W[U.index(u)] for u in U}

        return W, np.cumsum(timemarks)


class SFFSAValueIterationLeastSquares:

    def __init__(self,
                 env,
                 gpi,
                 constraint: Optional[dict] = None) -> None:

        self.env = env
        self.fsa = self.env.fsa
        self.gpi = gpi
        self.exit_states = self.env.exit_states
        self.constraint = constraint

    def traverse(self,
                 weights: Optional[dict] = None,
                 num_iters: Optional[dict] = 100) -> Union[dict, List[float]]:

        exit_states_mapping = self.exit_states

        U = self.fsa.states

        if weights is None:
            W = np.zeros((len(U), self.env.feat_dim))
            # For each fsa state we have a vector w of size phi that indicates
            # how important each goal is given that we're in that fsa state
        else:
            W = np.asarray(list(weights.values()))

        timemarks = [0]

        for _ in range(num_iters):

            start_time = time.time()

            W_old = W.copy()  # Keep to compare diff with new weights for stopping iteration

            # For each possible starting state
            for (uidx, u) in enumerate(U):

                if self.fsa.is_terminal(u):
                    continue

                all_phis = []
                all_q_targets = []

                for (vidx, v) in enumerate(U):
                    # For each transition
                    # i.e. for each other state v for which there exists an edge between u and v (transition)
                    if not self.fsa.graph.has_edge(u, v):
                        continue

                    # Get the predicate that satisfies the transition
                    proposition = self.fsa.get_predicate((u, v))
                    assert len(proposition) == 1, "Only one predicate is supported"
                    proposition = proposition[0]

                    exit_states = exit_states_mapping[self.fsa.symbols_to_phi[proposition]]  # This is a set

                    for exit_state in exit_states:
                        if self.fsa.is_terminal(v):
                            q_val = 1.0  # Terminal state
                        else:
                            psi = self.gpi.max_q(exit_state, W[vidx])
                            q_val = np.dot(psi, W[vidx])  # Q-value estimate

                        phi = self.env.env.features(state=None, action=None, next_state=exit_state)

                        all_q_targets.append(q_val)
                        all_phis.append(phi)

                phis = np.array(all_phis)
                q_targets = np.array(all_q_targets)

                # Do linear (regression) to fit w
                eps = 1e-5  # eps is defined as regularization parameter

                # Compute the regularized normal equation solution:
                # w = (Phi^T * Phi + eps * I)^(-1) * Phi^T * q_targets
                w = np.linalg.inv(phis.T @ phis + eps * np.eye(phis.shape[1])) @ (phis.T @ q_targets)

                # Could use np.linalg.solve for numerical stability:
                # A = phis.T @ phis + eps * np.eye(phis.shape[1])
                # b = phis.T @ q_targets
                # w = np.linalg.solve(A, b)

                W[uidx] = w

            elapsed_time = time.time() - start_time
            timemarks.append(elapsed_time)

            if np.allclose(W, W_old):
                break

        W = {u: W[U.index(u)] for u in U}

        return W, np.cumsum(timemarks)


def get_augmented_phi(phi, uidx, n_fsa_states, feat_dim):
    augmented_phi = np.zeros((n_fsa_states * feat_dim,))
    augmented_phi[uidx * feat_dim: (uidx + 1) * feat_dim] = phi
    return augmented_phi


def get_augmented_psis(psis, n_fsa_states, feat_dim, indicator_edge_has_proposition):
    augmented_psis = np.zeros((psis.shape[0], n_fsa_states * feat_dim))

    for i, psi in enumerate(psis):
        # Repeat psi across n_fsa_states times
        augmented_psi = np.tile(psi, n_fsa_states)
        augmented_psi *= indicator_edge_has_proposition
        augmented_psis[i, :] = augmented_psi
    return augmented_psis


class SFFSAValueIterationAugmented:

    def __init__(self,
                 env,
                 gpi,
                 constraint: Optional[dict] = None) -> None:

        self.env = env
        self.fsa = self.env.fsa
        self.gpi = gpi
        self.exit_states = self.env.exit_states
        self.all_exit_states = []
        self.constraint = constraint

        for key, exit_states in self.exit_states.items():
            self.all_exit_states.extend(list(exit_states))

        self.U = self.fsa.states
        self.n_exit_states = len(self.all_exit_states)
        self.n_fsa_states = len(self.U)
        self.feat_dim = self.env.feat_dim
        self.augmented_feature_dim = self.n_fsa_states * self.env.feat_dim
        self.curr_iter = 0

        self.indicator_edge_has_proposition = np.zeros((self.n_fsa_states, self.augmented_feature_dim))

        # For each possible starting state
        for (uidx, u) in enumerate(self.U):
            if self.fsa.is_terminal(u):
                continue

            indicator_edge_has_proposition = np.zeros((self.augmented_feature_dim,))

            for (vidx, v) in enumerate(self.U):
                if not self.fsa.graph.has_edge(u, v):
                    continue

                # Get the predicate that satisfies the transition
                proposition = self.fsa.get_predicate((u, v))
                assert len(proposition) == 1, "Only one predicate is supported"
                proposition = proposition[0]

                indicator_edge_has_proposition[vidx * self.feat_dim: (vidx + 1) * self.feat_dim] = (
                        np.array(self.env.env.prop_at_feat_idx) == proposition
                ).astype(int)

            self.indicator_edge_has_proposition[uidx, :] = indicator_edge_has_proposition

        for policy in self.gpi.policies:
            policy.set_augmented_psi_table(self.n_fsa_states, self.feat_dim, self.indicator_edge_has_proposition)

    def traverse(self,
                 weights=None,
                 num_iters=100):

        if weights is None:
            W = np.zeros((self.augmented_feature_dim,))
        else:
            W = np.asarray(list(weights.values())).reshape(-1)

        timemarks = [0]

        for _ in range(num_iters):
            start_time = time.time()

            W_old = W.copy()  # Keep to compare diff with new weights for stopping iteration

            all_augmented_phis = np.zeros((self.n_exit_states * self.n_fsa_states, self.augmented_feature_dim))
            all_q_targets = np.zeros((self.n_exit_states * self.n_fsa_states,))

            # For each possible starting state
            for (uidx, u) in enumerate(self.U):
                augmented_phis = np.zeros((self.n_exit_states, self.augmented_feature_dim))
                q_targets = []

                for i, exit_state in enumerate(self.all_exit_states):
                    if not self.fsa.is_terminal(u):
                        action, policy_index, q_target = self.gpi.eval(exit_state, W, uidx=uidx, return_q_val=True,
                                                                       return_policy_index=True)
                        # all_policy_q_vals = []
                        # for policy in self.gpi.policies:
                        #     psis = policy.q_table[exit_state]
                        #     augmented_psis = get_augmented_psis(psis, self.n_fsa_states, self.feat_dim,
                        #                                         self.indicator_edge_has_proposition[uidx])
                        #     augmented_psis_2 = policy.get_augmented_psis(uidx, exit_state)
                        #     np.testing.assert_array_equal(augmented_psis, augmented_psis_2)
                        #
                        #     q_vals = augmented_psis @ W
                        #
                        #     # Assuming greedy policies, expectation over actions becomes the maximum q-val, otherwise
                        #     # we should take something like weighted average using softmax over q-vals
                        #     max_q_val = np.max(q_vals)
                        #     all_policy_q_vals.append(max_q_val)
                        #
                        # # Maximize over policies
                        # q_target = max(all_policy_q_vals)

                        q_targets.append(q_target)
                    else:
                        q_targets.append(1)

                    phi = self.env.env.features(state=None, action=None, next_state=exit_state)
                    augmented_phi = get_augmented_phi(phi, uidx, self.n_fsa_states, self.feat_dim)
                    augmented_phis[i, :] = augmented_phi

                q_targets = np.array(q_targets)

                all_augmented_phis[uidx * self.n_exit_states: (uidx + 1) * self.n_exit_states, :] = augmented_phis
                all_q_targets[uidx * self.n_exit_states: (uidx + 1) * self.n_exit_states] = q_targets

            # Do linear (regression) to fit w
            eps = 1e-5  # eps is defined as regularization parameter

            # Compute the regularized normal equation solution:
            # w = (Phi^T * Phi + eps * I)^(-1) * Phi^T * q_targets
            W = np.linalg.inv(all_augmented_phis.T @ all_augmented_phis + eps * np.eye(all_augmented_phis.shape[1])) @ (all_augmented_phis.T @ all_q_targets)

            elapsed_time = time.time() - start_time
            timemarks.append(elapsed_time)

            if np.allclose(W, W_old):
                break

        self.curr_iter += 1

        W = {u: W[self.U.index(u) * self.feat_dim: (self.U.index(u) + 1) * self.feat_dim] for u in self.U}

        return W, np.cumsum(timemarks)
