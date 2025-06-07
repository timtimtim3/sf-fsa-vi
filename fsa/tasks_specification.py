from .fsa import FiniteStateAutomaton
import numpy as np


def load_fsa(name: str, env, fsa_symbols_from_env=False, using_lof=None):
    symbols_to_phi = None
    if fsa_symbols_from_env:
        symbols = env.PHI_OBJ_TYPES
        symbols_to_phi = {symbol: i for i, symbol in enumerate(symbols)}

    fsa_name = name
    kwargs = {}

    print(name)

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
    elif "detour" in name:
        fsa_name = "OfficeAreas-v0-Detour-task1"
        init_fun = fsa_detour
    elif "OfficeAreas" in name and "v0-task" in name:

        if "task1" in name:
            fsa_name = "Office-v0-task1"
            init_fun = fsa_officeAreas1
        elif "task2" in name:
            fsa_name = "Office-v0-task2"
            init_fun = fsa_officeAreas2
        elif "task3" in name:
            fsa_name = "Office-v0-task3"
            init_fun = fsa_officeAreas3
        elif "task4" in name:
            fsa_name = "Office-v0-task4"
            init_fun = fsa_officeAreas4
        elif "task5" in name:
            fsa_name = "Office-v0-task5"
            init_fun = fsa_officeAreas5
        elif "task6" in name:
            fsa_name = "Office-v0-task6"
            init_fun = fsa_officeAreas6

        if using_lof is not None:
            kwargs["using_lof"] = using_lof

    elif "Office" in name and "teleport" in name:
                
        if "teleport-task1" in name:
            fsa_name = "Office-v0-teleport-task1"
            init_fun = fsa_A_THEN_B
        elif "teleport-task2" in name:
            fsa_name = "Office-v0-teleport-task2"
            init_fun = fsa_A_OR_B
        elif "teleport-task3" in name:
            fsa_name = "Office-v0-teleport-task3"
            init_fun = fsa_A_AND_B
            
    elif name == "OfficeAreasRBFOnly-v0-SemiCircle-task1":
        init_fun = fsa_A_THEN_B
    else:
        raise NameError()

    if symbols_to_phi is not None:
        kwargs["symbols_to_phi"] = symbols_to_phi
    g = init_fun(env, fsa_name=fsa_name, **kwargs)
    
    return g

def fsa_pickup_dropoff1(env, fsa_name="fsa"):
    symbols_to_phi = {"H": 0, 
                      "C": 1,
                      "A": 2,
                      "T": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)
    
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

def fsa_pickup_dropoff2(env, fsa_name="fsa"):
    symbols_to_phi = {"H": 0, 
                      "C": 1,
                      "A": 2,
                      "T": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)
    
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

def fsa_pickup_dropoff3(env, fsa_name="fsa"):
    
    symbols_to_phi = {"H": 0, 
                      "C": 1,
                      "A": 2,
                      "T": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)
    
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


def fsa_double_slit1(env, fsa_name="fsa"):

    symbols_to_phi = {"O1": 0, 
                      "O2": 1,}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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

def fsa_delivery1(env, fsa_name="fsa"):

    # Sequential: Go to A, then B, then C, then H.
    # A -> B -> C -> H

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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

def fsa_delivery2(env, fsa_name="fsa"):

    # OR: Go to A "OR" B, then C, then H.
    # (A v B ) -> C -> H

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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

def fsa_delivery3(env, fsa_name="fsa"):

    # Composed: Go to A "AND" B in any order, then C, then H.
    # (A ^ B) -> C -> H

    symbols_to_phi = {"A": 0, 
                      "B": 1, 
                      "C": 2, 
                      "H": 3}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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


def set_exit_states_to_single(exit_states):
    simplified = {}
    for k, v in exit_states.items():
        if isinstance(v, set) and len(v) == 1:
            simplified[k] = next(iter(v))  # extract the single element
        else:
            simplified[k] = v
    return simplified


def fsa_office1(env, fsa_name="fsa"):

    # Sequential: Get coffee, then email, then office.
    # COFFEE -> MAIL -> OFFICE

    # or task (Coffee, then mail, then office)
    symbols_to_phi = {"C1": 0, 
                      "C2": 1, 
                      "O1": 2, 
                      "O2": 3,
                      "M1": 4,
                      "M2": 5}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u1", "u2", ["M1", "M2"])
    fsa.add_transition("u2", "u3", ["O1", "O2"])

    exit_states = set_exit_states_to_single(env.exit_states)
    exit_states_idxs = list(map(lambda x: env.coords_to_state[exit_states[x]], range(len(exit_states))))

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

def fsa_office2(env, fsa_name="fsa"):

    # OR: Get coffee OR email, then office.
    # (COFFEE v MAIL) -> OFFICE

    symbols_to_phi = {"C1": 0, 
                      "C2": 1, 
                      "O1": 2, 
                      "O2": 3,
                      "M1": 4,
                      "M2": 5}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["C1", "C2"])
    fsa.add_transition("u0", "u2", ["M1", "M2"])
    fsa.add_transition("u1", "u3", ["O1", "O2"])
    fsa.add_transition("u2", "u3", ["O1", "O2"])

    exit_states = set_exit_states_to_single(env.exit_states)
    exit_states_idxs = list(map(lambda x: env.coords_to_state[exit_states[x]], range(len(exit_states))))

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


def fsa_office3(env, fsa_name="fsa"):

    # Composite: Get mail AND coffee in any order, then go to an office
    # (COFEE ^ MAIL) -> OFFICE

    symbols_to_phi = {"C1": 0, 
                      "C2": 1, 
                      "O1": 2, 
                      "O2": 3,
                      "M1": 4,
                      "M2": 5}
    
    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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

    exit_states = set_exit_states_to_single(env.exit_states)
    exit_states_idxs = list(map(lambda x: env.coords_to_state[exit_states[x]], range(len(exit_states))))

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


def concat_same_kind_symbols(symbols):
    new_symbols = []
    kind_index = {}            # kind → index in new_symbols
    orig_to_new = {}           # original index → new index

    for i, symbol_list in enumerate(symbols):
        sym = symbol_list[0]   # e.g. "C1" or "C2" or "M"
        kind = sym[0]          # first character

        if kind in kind_index:
            idx = kind_index[kind]
            new_symbols[idx].append(sym)
        else:
            idx = len(new_symbols)
            kind_index[kind] = idx
            new_symbols.append([sym])

        orig_to_new[i] = idx

    return new_symbols, orig_to_new


def get_exit_states_idxs(env, orig_to_new, using_lof):
    exit_states_idxs = {}
    for proposition_idx, exit_states_set in env.exit_states.items():
        proposition_idx = proposition_idx if not using_lof else orig_to_new[proposition_idx]
        if proposition_idx not in exit_states_idxs:
            exit_states_idxs[proposition_idx] = set()

        for exit_state in exit_states_set:
            exit_state_idx = env.coords_to_state[exit_state]
            exit_states_idxs[proposition_idx].add(exit_state_idx)
    return exit_states_idxs


def fsa_officeAreas1(env, symbols_to_phi=None, fsa_name="fsa", using_lof=False):
    # Sequential: Go to A, then B, then C.
    # A -> B -> C

    if symbols_to_phi is None:
        symbols_to_phi = {"A": 0, "B": 1, "C": 2}
    symbols = list(symbols_to_phi.keys())  # e.g. ["A", "B", "C"]
    symbols = [[s] for s in symbols]  # results in [["A"], ["B"], ["C"]]
    if using_lof:
        symbols, orig_to_new = concat_same_kind_symbols(symbols)
    else:
        orig_to_new = None

    exit_states_idxs = get_exit_states_idxs(env, orig_to_new, using_lof)

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", symbols[0])
    fsa.add_transition("u1", "u2", symbols[1])
    fsa.add_transition("u2", "u3", symbols[2])

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

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


def fsa_officeAreas2(env, symbols_to_phi=None, fsa_name="fsa", using_lof=False):
    # OR: Get coffee OR email, then office.
    # (COFFEE v MAIL) -> OFFICE

    if symbols_to_phi is None:
        symbols_to_phi = {"A": 0, "B": 1, "C": 2}
    symbols = list(symbols_to_phi.keys())  # e.g. ["A", "B", "C"]
    symbols = [[s] for s in symbols]  # results in [["A"], ["B"], ["C"]]
    if using_lof:
        symbols, orig_to_new = concat_same_kind_symbols(symbols)
    else:
        orig_to_new = None

    exit_states_idxs = get_exit_states_idxs(env, orig_to_new, using_lof)

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", symbols[0])  # C
    fsa.add_transition("u0", "u2", symbols[1])  # M
    fsa.add_transition("u1", "u3", symbols[2])  # O
    fsa.add_transition("u2", "u3", symbols[2])  # O

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1

    for exit_state_idx in exit_states_idxs[1]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[0, 2, exit_state_idx] = 1  # Then we transition to u2

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[1, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u2 to u2 in all cases
    T[2, 2, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[2, 2, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[2, 3, exit_state_idx] = 1  # Then we transition to u3

    # Stay in the terminal state u3
    T[3, 3, :] = 1

    return fsa, T


def fsa_officeAreas3(env, symbols_to_phi=None, fsa_name="fsa", using_lof=False):
    # OR: Get coffee AND email, then office.
    # (COFFEE ^ MAIL) -> OFFICE

    if symbols_to_phi is None:
        symbols_to_phi = {"A": 0, "B": 1, "C": 2}
    symbols = list(symbols_to_phi.keys())  # e.g. ["A", "B", "C"]
    symbols = [[s] for s in symbols]  # results in [["A"], ["B"], ["C"]]
    if using_lof:
        symbols, orig_to_new = concat_same_kind_symbols(symbols)
    else:
        orig_to_new = None

    exit_states_idxs = get_exit_states_idxs(env, orig_to_new, using_lof)

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_transition("u0", "u1", symbols[0])  # C
    fsa.add_transition("u0", "u2", symbols[1])  # M
    fsa.add_transition("u1", "u3", symbols[1])  # M
    fsa.add_transition("u2", "u3", symbols[0])  # C
    fsa.add_transition("u3", "u4", symbols[2])  # O

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1

    for exit_state_idx in exit_states_idxs[1]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[0, 2, exit_state_idx] = 1  # Then we transition to u2

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[1, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u2 to u2 in all cases
    T[2, 2, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[2, 2, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[2, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u3 to u3 in all cases
    T[3, 3, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[3, 3, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[3, 4, exit_state_idx] = 1  # Then we transition to u4

    # Stay in the terminal state u4
    T[4, 4, :] = 1

    return fsa, T

def fsa_officeAreas4(env, symbols_to_phi=None, fsa_name="fsa", using_lof=False):
    # Sequential: Go to A, then B, then C, then B, then A
    # A -> B -> C -> B -> C

    if symbols_to_phi is None:
        symbols_to_phi = {"A": 0, "B": 1, "C": 2}
    symbols = list(symbols_to_phi.keys())  # e.g. ["A", "B", "C"]
    symbols = [[s] for s in symbols]  # results in [["A"], ["B"], ["C"]]
    if using_lof:
        symbols, orig_to_new = concat_same_kind_symbols(symbols)
    else:
        orig_to_new = None

    exit_states_idxs = get_exit_states_idxs(env, orig_to_new, using_lof)

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")
    fsa.add_state("u5")

    fsa.add_transition("u0", "u1", symbols[0])
    fsa.add_transition("u1", "u2", symbols[1])
    fsa.add_transition("u2", "u3", symbols[2])
    fsa.add_transition("u3", "u4", symbols[1])
    fsa.add_transition("u4", "u5", symbols[0])

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

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

    # Transition from u3 to u3 in all cases
    T[3, 3, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[3, 3, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area B
        T[3, 4, exit_state_idx] = 1  # Then we transition to u4

    # Transition from u4 to u4 in all cases
    T[4, 4, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[4, 4, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area A
        T[4, 5, exit_state_idx] = 1  # Then we transition to u5

    # Stay in the terminal state u5
    T[5, 5, :] = 1

    return fsa, T

def fsa_officeAreas5(env, symbols_to_phi=None, fsa_name="fsa", using_lof=False):
    # OR: Get coffee OR email, then office.
    # (COFFEE v MAIL) -> OFFICE -> MAIL -> (COFFEE or OFFICE)

    if symbols_to_phi is None:
        symbols_to_phi = {"A": 0, "B": 1, "C": 2}
    symbols = list(symbols_to_phi.keys())  # e.g. ["A", "B", "C"]
    symbols = [[s] for s in symbols]  # results in [["A"], ["B"], ["C"]]
    if using_lof:
        symbols, orig_to_new = concat_same_kind_symbols(symbols)
    else:
        orig_to_new = None

    exit_states_idxs = get_exit_states_idxs(env, orig_to_new, using_lof)

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")
    fsa.add_state("u5")
    fsa.add_state("u6")

    fsa.add_transition("u0", "u1", symbols[0])  # C
    fsa.add_transition("u0", "u2", symbols[1])  # M
    fsa.add_transition("u1", "u3", symbols[2])  # O
    fsa.add_transition("u2", "u3", symbols[2])  # O
    fsa.add_transition("u3", "u4", symbols[1])  # M
    fsa.add_transition("u4", "u5", symbols[0])  # C
    fsa.add_transition("u4", "u6", symbols[2])  # O

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1
    for exit_state_idx in exit_states_idxs[1]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[0, 2, exit_state_idx] = 1  # Then we transition to u2

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[1, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u2 to u2 in all cases
    T[2, 2, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[2, 2, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[2, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u3 to u3 in all cases
    T[3, 3, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[3, 3, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[3, 4, exit_state_idx] = 1  # Then we transition to u4

    # Transition from u4 to u4 in all cases
    T[4, 4, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[4, 4, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[4, 5, exit_state_idx] = 1  # Then we transition to u5
    for exit_state_idx in exit_states_idxs[2]:
        T[4, 4, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[4, 6, exit_state_idx] = 1  # Then we transition to u6

    # Stay in the terminal states u5 and u6
    T[5, 5, :] = 1
    T[6, 6, :] = 1

    return fsa, T

def fsa_officeAreas6(env, symbols_to_phi=None, fsa_name="fsa", using_lof=False):
    # OR: Get coffee AND email, then office,
    # (COFFEE ^ MAIL) -> OFFICE -> (COFFEE ^ MAIL)

    if symbols_to_phi is None:
        symbols_to_phi = {"A": 0, "B": 1, "C": 2}
    symbols = list(symbols_to_phi.keys())  # e.g. ["A", "B", "C"]
    symbols = [[s] for s in symbols]  # results in [["A"], ["B"], ["C"]]
    if using_lof:
        symbols, orig_to_new = concat_same_kind_symbols(symbols)
    else:
        orig_to_new = None

    exit_states_idxs = get_exit_states_idxs(env, orig_to_new, using_lof)

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")
    fsa.add_state("u4")

    fsa.add_state("u5")
    fsa.add_state("u6")
    fsa.add_state("u7")
    fsa.add_state("u8")

    fsa.add_transition("u0", "u1", symbols[0])  # C
    fsa.add_transition("u0", "u2", symbols[1])  # M
    fsa.add_transition("u1", "u3", symbols[1])  # M
    fsa.add_transition("u2", "u3", symbols[0])  # C
    fsa.add_transition("u3", "u4", symbols[2])  # O

    fsa.add_transition("u4", "u5", symbols[0])  # C
    fsa.add_transition("u4", "u6", symbols[1])  # M
    fsa.add_transition("u5", "u7", symbols[1])  # M
    fsa.add_transition("u6", "u8", symbols[0])  # C

    T = np.zeros((len(fsa.states), len(fsa.states), env.s_dim))

    # Transition from u0 to u0 in all cases
    T[0, 0, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[0, 1, exit_state_idx] = 1  # Then we transition to u1

    for exit_state_idx in exit_states_idxs[1]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[0, 2, exit_state_idx] = 1  # Then we transition to u2

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[1, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u2 to u2 in all cases
    T[2, 2, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[2, 2, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[2, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u3 to u3 in all cases
    T[3, 3, :] = 1
    for exit_state_idx in exit_states_idxs[2]:
        T[3, 3, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area O
        T[3, 4, exit_state_idx] = 1  # Then we transition to u4

    # Transition from u4 to u4 in all cases
    T[4, 4, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[4, 4, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[4, 5, exit_state_idx] = 1  # Then we transition to u5

    for exit_state_idx in exit_states_idxs[1]:
        T[4, 4, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[4, 6, exit_state_idx] = 1  # Then we transition to u6

    # Transition from u5 to u5 in all cases
    T[5, 5, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[5, 5, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area M
        T[5, 7, exit_state_idx] = 1  # Then we transition to u7

    # Transition from u6 to u6 in all cases
    T[6, 6, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[6, 6, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area C
        T[6, 8, exit_state_idx] = 1  # Then we transition to u8

    # Stay in the terminal state u7 and u8
    T[7, 7, :] = 1
    T[8, 8, :] = 1

    return fsa, T

def fsa_A_THEN_B(env, fsa_name="fsa"):
    # Sequential: Go to A, then B
    # A -> B

    symbols_to_phi = {"A": 0,
                      "B": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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

def fsa_A_OR_B(env, fsa_name="fsa"):
    # Disjunctive: Go to A OR B
    # A -> B

    symbols_to_phi = {"A": 0,
                      "B": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u0", "u2", ["B"])

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
    for exit_state_idx in exit_states_idxs[1]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area B
        T[0, 2, exit_state_idx] = 1  # Then we transition to u2

    # Stay in the terminal state u1
    T[1, 1, :] = 1

    # Stay in the terminal state u2
    T[2, 2, :] = 1

    return fsa, T

def fsa_A_AND_B(env, fsa_name="fsa"):
    # Conjunctive: Go to A AND B in any order
    # A -> B

    symbols_to_phi = {"A": 0,
                      "B": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

    fsa.add_state("u0")
    fsa.add_state("u1")
    fsa.add_state("u2")
    fsa.add_state("u3")

    fsa.add_transition("u0", "u1", ["A"])
    fsa.add_transition("u0", "u2", ["B"])
    fsa.add_transition("u1", "u3", ["B"])
    fsa.add_transition("u2", "u3", ["A"])

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
    for exit_state_idx in exit_states_idxs[1]:
        T[0, 0, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area B
        T[0, 2, exit_state_idx] = 1  # Then we transition to u2

    # Transition from u1 to u1 in all cases
    T[1, 1, :] = 1
    for exit_state_idx in exit_states_idxs[1]:
        T[1, 1, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area B
        T[1, 3, exit_state_idx] = 1  # Then we transition to u3

    # Transition from u2 to u2 in all cases
    T[2, 2, :] = 1
    for exit_state_idx in exit_states_idxs[0]:
        T[2, 2, exit_state_idx] = 0  # Except if we are in some exit state tile located in Area A
        T[2, 3, exit_state_idx] = 1  # Then we transition to u3

    # Stay in the terminal state u3
    T[3, 3, :] = 1

    return fsa, T

def fsa_detour(env, fsa_name="fsa"):
    # Sequential: Go to A

    symbols_to_phi = {"A": 0,
                      "B": 1}

    fsa = FiniteStateAutomaton(symbols_to_phi, fsa_name=fsa_name)

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