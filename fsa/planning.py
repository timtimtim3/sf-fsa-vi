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
                        # If the transition is terminal we get a reward/w of 1 for each proposition that enables this
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