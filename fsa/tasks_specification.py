from .fsa import FiniteStateAutomaton
import numpy as np


def load_fsa(name: str, env):
    if name == "PickupDropoff-v0-task1":    
        init_fun = fsa_pickup_dropoff1
    elif name == "PickupDropoff-v0-task2":    
        init_fun = fsa_pickup_dropoff2
    elif name == "PickupDropoff-v0-task3":    
        init_fun = fsa_pickup_dropoff3
    elif name == "Delivery-v0-task1":
        init_fun = fsa_delivery1
    elif name == "Delivery-v0-task2":
        init_fun = fsa_delivery2
    elif name == "Delivery-v0-task3":
        init_fun = fsa_delivery3
    elif name == "Office-v0-task1":
        init_fun = fsa_office1
    elif name == "Office-v0-task2":
        init_fun = fsa_office2
    elif name == "Office-v0-task3":
        init_fun = fsa_office3
    elif name == "DoubleSlit-v0-task1":
        init_fun = fsa_double_slit1
    elif name == "OfficeAreas-v0-task1" or name == "OfficeAreasRBF-v0-task1" or name == "OfficeAreasRBFOnly-v0-task1" \
            or name == "OfficeAreasFeatures-v0-task1":
        init_fun = fsa_officeAreas1
    elif name == "OfficeAreasRBFOnly-v0-SemiCircle-task1":
        init_fun = fsa_officeAreasSemiCircle1
    elif name == "OfficeAreasFeatures-v0-detour":
        init_fun = fsa_detour
    else:
        raise NameError()
    
    g = init_fun(env)
    
    return g

def fsa_pickup_dropoff1(env):
    symbols_to_phi = {"H": 0, 
                      "C": 1,
                      "A": 2,
                      "T": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)
    
    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u1", "u2", ["H"])
    fsa.add_transition("u2", "u3", ["T"])
    fsa.add_transition("u3", "u4", ["C"])
    
    exit_states_idxs = list(map(lambda x: env.coords_to_state[x], env.exit_states.values())) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # From u0 to u1 (via (A)irport)
    T[0, 0, :] = 1
    T[0, 0, exit_states_idxs[2]] = 0
    T[0, 1, exit_states_idxs[2]] = 1
    
    # From u1 to u2 (via (H)otel)
    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[0]] = 0
    T[1, 2, exit_states_idxs[0]] = 1
    
    # From u2 to u3 (via (T)rain station)
    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[3]] = 0
    T[2, 3, exit_states_idxs[3]] = 1
    
    # From u3 to u4 (via (C)onvention center)
    T[3, 3, :] = 1
    T[3, 3, exit_states_idxs[1]] = 0
    T[3, 4, exit_states_idxs[1]] = 1

    T[4, 4, :] = 1

    return fsa, T

def fsa_pickup_dropoff2(env):
    symbols_to_phi = {"H": 0, 
                      "C": 1,
                      "A": 2,
                      "T": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)
    
    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u0", "u2", ["T"])
    fsa.add_transition("u1", "u3", ["H"])
    fsa.add_transition("u2", "u3", ["H"])
    
    exit_states_idxs = list(map(lambda x: env.coords_to_state[x], env.exit_states.values())) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via (A)irport or (T)rain station)
    T[0, 0, :] = 1

    # From u0 to u1 (A)irport 
    T[0, 0, exit_states_idxs[2]] = 0
    T[0, 1, exit_states_idxs[2]] = 1

    # From u1 to u2 (T)rain station
    T[0, 0, exit_states_idxs[3]] = 0
    T[0, 1, exit_states_idxs[3]] = 1
    
    # From u1 to u3 via (H)otel
    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[0]] = 0
    T[1, 3, exit_states_idxs[0]] = 1

    # From u2 to u3 via (H)otel
    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[0]] = 0
    T[2, 3, exit_states_idxs[0]] = 1
    
    T[3, 3, :] = 1

    return fsa, T

def fsa_pickup_dropoff3(env):
    
    symbols_to_phi = {"H": 0, 
                      "C": 1,
                      "A": 2,
                      "T": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)
    
    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")
    fsa.add_state("u5")
    fsa.add_state("u6")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u0", "u2", ["T"])
    fsa.add_transition("u1", "u3", ["H"])
    fsa.add_transition("u2", "u3", ["H"])
    fsa.add_transition("u3", "u4", ["A"])
    fsa.add_transition("u3", "u5", ["T"])
    fsa.add_transition("u4", "u6", ["C"])
    fsa.add_transition("u5", "u6", ["C"])

    exit_states_idxs = list(map(lambda x: env.coords_to_state[x], env.exit_states.values())) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via (A)irport or (T)rain station)
    T[0, 0, :] = 1
    # (A)irport 
    T[0, 0, exit_states_idxs[2]] = 0
    T[0, 1, exit_states_idxs[2]] = 1
    # (T)rain station
    T[0, 0, exit_states_idxs[3]] = 0
    T[0, 2, exit_states_idxs[3]] = 1
    
    # From u1 to u3 via (H)otel
    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[0]] = 0
    T[1, 3, exit_states_idxs[0]] = 1

    # From u2 to u3 via (H)otel
    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[0]] = 0
    T[2, 3, exit_states_idxs[0]] = 1
    
    # From u3 to u4 via (A)irport
    T[3, 3, :] = 1
    T[3, 3, exit_states_idxs[2]] = 0
    T[3, 4, exit_states_idxs[2]] = 1

    # From u3 to u5 via (T)rain station
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 5, exit_states_idxs[3]] = 1

    # From u4 to u6 via (C)onvention center
    T[4, 4, :] = 1
    T[4, 4, exit_states_idxs[1]] = 0
    T[4, 6, exit_states_idxs[1]] = 1
    
    # From u5 to u6 via (C)onvention center
    T[5, 5, :] = 1
    T[5, 5, exit_states_idxs[1]] = 0
    T[5, 6, exit_states_idxs[1]] = 1

    T[6, 6, :] = 1


    return fsa, T


def fsa_double_slit1(env):

    symbols_to_phi = {"O1": 0, 
                      "O2": 1,}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")

    fsa.add_transition("u0", "u1", "O1 v O2")

    exit_states_idxs = list(map(lambda x: env.coords_to_state[x], env.exit_states.values())) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via any coffee) or u2 (via any mail)
    T[0, 0, :] = 1

    # If it goes to an office or mail location, it transitions to a new F-state 
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 1, exit_states_idxs[0]] = 1
    T[0, 1, exit_states_idxs[1]] = 1

    T[1, 1, :] = 1

    return fsa, T

def fsa_delivery1(env):

    # Sequential: Go to A, then B, then C, then H.
    # A -> B -> C -> H

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u1", "u2", "B")
    fsa.add_transition("u2", "u3", "C")
    fsa.add_transition("u3", "u4", "H")

    exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states)))) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))


    T[0, 0, :] = 1
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 1, exit_states_idxs[0]] = 1 

    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[1]] = 0
    T[1, 2, exit_states_idxs[1]] = 1

    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 3, exit_states_idxs[2]] = 1

    T[3, 3, :] = 1
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 4, exit_states_idxs[3]] = 1

    T[4, 4, :] = 1

    return fsa, T

def fsa_delivery2(env):

    # OR: Go to A "OR" B, then C, then H.
    # (A v B ) -> C -> H

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u0", "u2", "B")
    fsa.add_transition("u1", "u3", "C")
    fsa.add_transition("u2", "u3", "C")
    fsa.add_transition("u3", "u4", "H")

    exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states)))) 
    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via A) or u2 (via B)
    T[0, 0, :] = 1
    # If it goes to A or B, it transitions to a new F-state
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 1, exit_states_idxs[0]] = 1 
    T[0, 2, exit_states_idxs[1]] = 1 

    # From u1 to u3
    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[2]] = 0
    T[1, 3, exit_states_idxs[2]] = 1

    # From u2 to u3
    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 3, exit_states_idxs[2]] = 1
    
    # From u2 to u4
    T[3, 3, :] = 1
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 4, exit_states_idxs[3]] = 1

    T[4, 4, :] = 1

    return fsa, T

def fsa_delivery3(env):

    # Composed: Go to A "AND" B in any order, then C, then H.
    # (A ^ B) -> C -> H

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")
    fsa.add_state("u5")


    fsa.add_transition("u0", "u1", "A")
    fsa.add_transition("u0", "u2", "B")
    fsa.add_transition("u1", "u3", "B")
    fsa.add_transition("u2", "u3", "A")
    fsa.add_transition("u3", "u4", "C")
    fsa.add_transition("u4", "u5", "H")

    exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states)))) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via A)
    T[0, 0, :] = 1
    
    # If it goes to A or B, it transitions to a new F-state
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 1, exit_states_idxs[0]] = 1 
    T[0, 2, exit_states_idxs[1]] = 1 

    # From u1 to u3
    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[1]] = 0
    T[1, 3, exit_states_idxs[1]] = 1

    # From u2 to u3
    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[0]] = 0
    T[2, 3, exit_states_idxs[0]] = 1
    
    # From u3 to u4
    T[3, 3, :] = 1
    T[3, 3, exit_states_idxs[2]] = 0
    T[3, 4, exit_states_idxs[2]] = 1
    
    # From u4 to u5
    T[4, 4, :] = 1
    T[4, 4, exit_states_idxs[3]] = 0
    T[4, 5, exit_states_idxs[3]] = 1

    # From u4 to u5
    T[5, 5, :] = 1

    return fsa, T


def fsa_office1(env):

    # Sequential: Get coffee, then email, then office.
    # COFFEE -> MAIL -> OFFICE

    # or task (Coffee, then mail, then office)
    symbols_to_phi = {"C1": 0, 
                      "C2": 1, 
                      "O1": 2, 
                      "O2": 3,
                      "M1": 4,
                      "M2": 5}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u1", "u2", ["M1", "M2"])
    fsa.add_transition("u2", "u3", ["O1", "O2"])

    exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states))))

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via any coffee)
    T[0, 0, :] = 1
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 1, exit_states_idxs[0]] = 1
    T[0, 1, exit_states_idxs[1]] = 1

    # This is from u1 to u2 (via any mail)
    T[1, 1, :] = 1
    T[1, 1, exit_states_idxs[4]] = 0
    T[1, 1, exit_states_idxs[5]] = 0
    T[1, 2, exit_states_idxs[4]] = 1
    T[1, 2, exit_states_idxs[5]] = 1

    # This is from u2 to u3 (via any office)
    T[2, 2, :] = 1
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 2, exit_states_idxs[3]] = 0
    T[2, 3, exit_states_idxs[2]] = 1
    T[2, 3, exit_states_idxs[3]] = 1

    T[3, 3, :] = 1

    return fsa, T

def fsa_office2(env):

    # OR: Get coffee OR email, then office.
    # (COFFEE v MAIL) -> OFFICE

    symbols_to_phi = {"C1": 0, 
                      "C2": 1, 
                      "O1": 2, 
                      "O2": 3,
                      "M1": 4,
                      "M2": 5}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u0", "u2", ["M1", "M2"])
    fsa.add_transition("u1", "u3", ["O1", "O2"])
    fsa.add_transition("u2", "u3", ["O1", "O2"])

    exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states)))) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via any coffee) or u2 (via any mail)
    T[0, 0, :] = 1

    # If it goes to an office or mail location, it transitions to a new F-state 
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 1, exit_states_idxs[0]] = 1
    T[0, 1, exit_states_idxs[1]] = 1

    T[0, 0, exit_states_idxs[4]] = 0
    T[0, 0, exit_states_idxs[5]] = 0   
    T[0, 2, exit_states_idxs[4]] = 1
    T[0, 2, exit_states_idxs[5]] = 1

    # This is from u1 to u3 (via any mail)
    T[1, 1, :] = 1

    T[1, 1, exit_states_idxs[2]] = 0
    T[1, 1, exit_states_idxs[3]] = 0
    T[1, 3, exit_states_idxs[2]] = 1
    T[1, 3, exit_states_idxs[3]] = 1

    # This is from u2 to u3 (via any coffee)
    T[2, 2, :] = 1

    # If it goes to any coffee location, it transitions to a new F-state 
    T[2, 2, exit_states_idxs[2]] = 0
    T[2, 2, exit_states_idxs[3]] = 0
    T[2, 3, exit_states_idxs[2]] = 1
    T[2, 3, exit_states_idxs[3]] = 1

    T[3, 3, :] = 1

    return fsa, T


def fsa_office3(env):

    # Composite: Get mail AND coffee in any order, then go to an office
    # (COFEE ^ MAIL) -> OFFICE

    symbols_to_phi = {"C1": 0, 
                      "C2": 1, 
                      "O1": 2, 
                      "O2": 3,
                      "M1": 4,
                      "M2": 5}
    
    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u0", "u2", ["M1", "M2"])
    fsa.add_transition("u1", "u3", ["M1", "M2"])
    fsa.add_transition("u2", "u3", ["C1", "C2"])
    fsa.add_transition("u3", "u4", ["O1", "O2"])

    exit_states_idxs = list(map(lambda x: env.coords_to_state[env.exit_states[x]], range(len(env.exit_states)))) 

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # This is from u0 to u1 (via any coffee) or u2 (via any mail)
    T[0, 0, :] = 1

    # If it goes to an office or mail location, it transitions to a new F-state 
    T[0, 0, exit_states_idxs[0]] = 0
    T[0, 0, exit_states_idxs[1]] = 0
    T[0, 0, exit_states_idxs[4]] = 0
    T[0, 0, exit_states_idxs[5]] = 0
    T[0, 1, exit_states_idxs[0]] = 1
    T[0, 1, exit_states_idxs[1]] = 1
    T[0, 2, exit_states_idxs[4]] = 1
    T[0, 2, exit_states_idxs[5]] = 1

    # This is from u1 to u3 (via any mail)
    T[1, 1, :] = 1

    # If it goes to any mail location, it transitions to a new F-state 
    T[1, 1, exit_states_idxs[4]] = 0
    T[1, 1, exit_states_idxs[5]] = 0
    T[1, 2, exit_states_idxs[4]] = 1
    T[1, 2, exit_states_idxs[5]] = 1

    # This is from u2 to u3 (via any coffee)
    T[2, 2, :] = 1

    # If it goes to any coffee location, it transitions to a new F-state 
    T[2, 2, exit_states_idxs[0]] = 0
    T[2, 2, exit_states_idxs[1]] = 0
    T[2, 3, exit_states_idxs[0]] = 1
    T[2, 3, exit_states_idxs[1]] = 1

    # This is from u3 to u4 (terminal, via any office location)
    T[3, 3, :] = 1

    # If it goes to any coffee location, it transitions to a new F-state 
    T[3, 3, exit_states_idxs[2]] = 0
    T[3, 3, exit_states_idxs[3]] = 0
    T[3, 4, exit_states_idxs[2]] = 1
    T[3, 4, exit_states_idxs[3]] = 1

    T[4, 4, :] = 1

    return fsa, T


def fsa_officeAreas1(env):
    # Sequential: Go to A, then B, then C.
    # A -> B -> C

    symbols_to_phi = {"A": 0,
                      "B": 1,
                      "C": 2}

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u1", "u2", ["B"])
    fsa.add_transition("u2", "u3", ["C"])

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    exit_states_idxs = {}
    for proposition_idx, exit_states_set in env.exit_states.items():
        exit_states_idxs[proposition_idx] = set()
        for exit_state in exit_states_set:
            exit_state_idx = env.coords_to_state[exit_state]
            exit_states_idxs[proposition_idx].add(exit_state_idx)

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area A
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area B
        T[1, 2, exit_state_idx] = 1  # Then we transition to u2

    # Transition from u2 to u2 in all cases
    T[2, 2, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[2, 2, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[2, 3, exit_state_idx] = 1  # Then we transition to u3

    # Stay in the terminal state u3
    T[3, 3, :] = 1

    return fsa, T


def fsa_officeAreasSemiCircle1(env):
    # Sequential: Go to A, then B
    # A -> B

    symbols_to_phi = {"A": 0,
                      "B": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u1", "u2", ["B"])

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    exit_states_idxs = {}
    for proposition_idx, exit_states_set in env.exit_states.items():
        exit_states_idxs[proposition_idx] = set()
        for exit_state in exit_states_set:
            exit_state_idx = env.coords_to_state[exit_state]
            exit_states_idxs[proposition_idx].add(exit_state_idx)

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area A
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area B
        T[1, 2, exit_state_idx] = 1  # Then we transition to u2

    # Stay in the terminal state u2
    T[2, 2, :] = 1

    return fsa, T

def fsa_detour(env):
    # Sequential: Go to A

    symbols_to_phi = {"A": 0,
                      "B": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi)

    fsa.add_state("u0")
    fsa.add_state("u1")

    fsa.add_transition("u0", "u1", ["A"])

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    exit_states_idxs = {}
    for proposition_idx, exit_states_set in env.exit_states.items():
        exit_states_idxs[proposition_idx] = set()
        for exit_state in exit_states_set:
            exit_state_idx = env.coords_to_state[exit_state]
            exit_states_idxs[proposition_idx].add(exit_state_idx)

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area A
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1

    # Stay in the terminal state u1
    T[1, 1, :] = 1

    return fsa, T


if __name__ == "__main__":

    fsa, _ = fsa_office1()

    M = fsa.get_transition_matrix()

    print(M)