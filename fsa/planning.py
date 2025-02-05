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
                            # print(e)
                            w[idx] = np.dot(self.gpi.max_q(e, W[vidx]), W[vidx]) # Assign to each goal/predicate in w
                            # the maximum Q-value we can get in its corresponding exit state (over policies and actions)

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