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
        else:
            W = np.asarray(list(weights.values()))

        timemarks = [0]
        
        for _ in range(num_iters):

            start_time = time.time()

            W_old = W.copy()
            
            for (uidx, u) in enumerate(U):
                
                if self.fsa.is_terminal(u):
                    continue
                
                weights = []

                for (vidx, v) in enumerate(U):

                    if not self.fsa.graph.has_edge(u, v):
                        continue

                    w = np.zeros((self.env.feat_dim))

                    # Get the predicates satisfied by the transition
                    propositions = self.fsa.get_predicate((u, v)) 
                    idxs = [self.fsa.symbols_to_phi[prop] for prop in propositions]

                    if self.fsa.is_terminal(v): 
                        w[idxs] = 1
                    else:
                        for idx in idxs:

                            e = exit_states[idx]
                            w[idx] = np.dot(self.gpi.max_q(e, W[vidx]), W[vidx])

                    weights.append(w)                   
                
                weights = np.asarray(weights)
                weights = np.sum(weights, axis=0)
                
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