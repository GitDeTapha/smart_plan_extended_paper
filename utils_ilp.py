# utils_ilp.py
# Routines of the scheduling notebook (03_scheduling_ilp.ipynb).
#
# One clinic day of T slots; slot t lasts L[t] minutes. Patient i comes with
# probability q[i] and needs s[i] minutes. Each patient gets exactly one slot,
# a slot holds at most two patients, served shortest first (SPT). Each slot
# starts on time (the cost is separable by slot). Expected cost of the day:
#     Z = c_I * E[idle time] + c_W * E[waiting time].
# Every choice (durations, costs, solver options) is passed as an argument
# from the "Paramètres" cell of the notebook.

import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


############################################################
#                                                          #
#        ------  Expected cost of one slot  -------        #
#                                                          #
############################################################

def slot_cost_enumeration(patients, q, s, L):
    """
    Expected idle time and expected waiting time of ONE slot, by listing every
    show/no-show scenario of its patients (2 scenarios for one patient, 4 for
    two). Used to check the closed-form coefficients of the ILP.

    Args:
        patients (list): Patients of the slot (0, 1 or 2).
        q, s (arrays): Show probabilities and durations (minutes) of all patients.
        L (float): Length of the slot (minutes).

    Returns:
        tuple: (expected idle time, expected waiting time), in minutes.
    """
    e_idle = e_wait = 0.0
    for shows in itertools.product([True, False], repeat=len(patients)):
        prob = np.prod([q[i] if come else 1 - q[i] for i, come in zip(patients, shows)])
        present = sorted([i for i, come in zip(patients, shows) if come], key=lambda i: s[i])  # SPT
        clock = wait = 0.0
        for i in present:
            wait += clock          # a patient waits until the previous one is done
            clock += s[i]
        e_idle += prob * max(0.0, L - clock)
        e_wait += prob * wait
    return e_idle, e_wait


def planning_cost(planning, q, s, L, c_I, c_W):
    """
    Expected cost of a planning under the model (each slot starts on time).

    Args:
        planning (dict): {slot: list of patients}; a slot absent from the dict is empty.
        q, s (arrays): Show probabilities and durations.
        L (list): Slot lengths.
        c_I, c_W (float): Costs of one minute of idle time and of waiting time.

    Returns:
        dict: Z, E_idle and E_wait (in minutes, unweighted).
    """
    e_idle = e_wait = 0.0
    for t in range(len(L)):
        idle, wait = slot_cost_enumeration(planning.get(t, []), q, s, L[t])
        e_idle += idle
        e_wait += wait
    return {"Z": c_I * e_idle + c_W * e_wait, "E_idle": e_idle, "E_wait": e_wait}


############################################################
#                                                          #
#        ------        The ILP          -------            #
#                                                          #
############################################################

def build_milp(q, s, L, c_I, c_W):
    """
    Build the ILP of the paper as the vectors and matrices of scipy.optimize.milp:
        minimise c @ v   subject to   A_lb <= A @ v <= A_ub,  bounds on v,  some v integer.

    Variables (v is their concatenation):
        x[i, t] in {0, 1}   patient i is in slot t                         (n * T variables)
        z[i, j, t] >= 0     patients i < j are both in slot t (continuous) (n(n-1)/2 * T variables)

    Closed forms (one slot of length L, SPT order):
        E[idle] of the slot = L - sum_i a[i,t] x[i,t] - sum_{i<j} delta[i,j,t] z[i,j,t]
            a[i,t]       = q_i * min(s_i, L)                 expected busy time of i alone
            delta[i,j,t] = q_i q_j (min(s_i+s_j, L) - min(s_i, L) - min(s_j, L)) <= 0
                                                             correction when both come and exceed L
        E[wait] of the slot = sum_{i<j} w[i,j] z[i,j,t]
            w[i,j]       = q_i q_j min(s_i, s_j)              the longer one waits for the shorter one

    So Z = c_I * sum_t L_t + sum_{i,t} (-c_I a[i,t]) x[i,t] + sum_{i<j,t} (c_W w[i,j] - c_I delta[i,j,t]) z[i,j,t].
    The constant c_I * sum_t L_t is left out of c and added back to the result.

    Constraints:
        C1  sum_t x[i,t] = 1                      every patient gets exactly one slot
        C2  sum_i x[i,t] <= 2                     at most two patients per slot
        C3  z[i,j,t] >= x[i,t] + x[j,t] - 1       z = 1 when i and j share slot t
    The coefficient of z is >= 0 (delta <= 0), so the minimisation pushes z down
    to max(0, x_it + x_jt - 1), which is 0 or 1: z can be continuous (Proposition 1).

    Args:
        q, s (arrays): Show probabilities and durations (minutes).
        L (list): Slot lengths (minutes).
        c_I, c_W (float): Costs of one minute of idle time and of waiting time.

    Returns:
        dict: c, constraints, bounds, integrality, the variable index maps and the constant.
    """
    n, T = len(q), len(L)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    x_index = {(i, t): k for k, (i, t) in enumerate(itertools.product(range(n), range(T)))}
    z_index = {(i, j, t): len(x_index) + k for k, ((i, j), t) in enumerate(itertools.product(pairs, range(T)))}
    n_var = len(x_index) + len(z_index)

    c = np.zeros(n_var)
    for (i, t), k in x_index.items():
        c[k] = -c_I * q[i] * min(s[i], L[t])
    for (i, j, t), k in z_index.items():
        w = q[i] * q[j] * min(s[i], s[j])
        delta = q[i] * q[j] * (min(s[i] + s[j], L[t]) - min(s[i], L[t]) - min(s[j], L[t]))
        c[k] = c_W * w - c_I * delta
    assert c[len(x_index):].min() >= -1e-9, "a z coefficient is negative: Proposition 1 does not apply"

    rows, lower, upper = [], [], []
    for i in range(n):                                           # C1
        row = np.zeros(n_var)
        row[[x_index[i, t] for t in range(T)]] = 1
        rows.append(row); lower.append(1); upper.append(1)
    for t in range(T):                                           # C2
        row = np.zeros(n_var)
        row[[x_index[i, t] for i in range(n)]] = 1
        rows.append(row); lower.append(-np.inf); upper.append(2)
    for (i, j, t), k in z_index.items():                         # C3: x_it + x_jt - z_ijt <= 1
        row = np.zeros(n_var)
        row[[x_index[i, t], x_index[j, t]]] = 1
        row[k] = -1
        rows.append(row); lower.append(-np.inf); upper.append(1)

    integrality = np.zeros(n_var)
    integrality[:len(x_index)] = 1                               # x integer (binary with the bounds), z continuous
    return {"c": c, "constraints": LinearConstraint(np.array(rows), lower, upper),
            "bounds": Bounds(np.zeros(n_var), np.ones(n_var)), "integrality": integrality,
            "x_index": x_index, "z_index": z_index, "constant": c_I * sum(L)}


def solve_ilp(q, s, L, c_I, c_W, time_limit, mip_rel_gap):
    """
    Solve the ILP with scipy.optimize.milp (HiGHS: dual simplex for each linear
    relaxation, branch-and-bound on the x variables).

    Args:
        q, s (arrays): Show probabilities and durations (minutes).
        L (list): Slot lengths (minutes).
        c_I, c_W (float): Costs of one minute of idle time and of waiting time.
        time_limit (float): Seconds after which HiGHS stops.
        mip_rel_gap (float): HiGHS stops when (best planning - lower bound) / |best planning| is below this.

    Returns:
        dict: planning {slot: patients}, Z (recomputed by enumeration), status
        (True if proven optimal), gap, number of branch-and-bound nodes, time (s).
    """
    if len(q) > 2 * len(L):
        raise ValueError(f"{len(q)} patients cannot fit in {len(L)} slots of two patients")
    model = build_milp(q, s, L, c_I, c_W)
    start = time.perf_counter()
    res = milp(model["c"], constraints=model["constraints"], bounds=model["bounds"],
               integrality=model["integrality"],
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False})
    seconds = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(f"HiGHS found no planning: {res.message}")
    planning = {}
    for (i, t), k in model["x_index"].items():
        if res.x[k] > 0.5:
            planning.setdefault(t, []).append(i)
    cost = planning_cost(planning, q, s, L, c_I, c_W)
    # The cost of the planning recomputed by enumeration must equal the solver's objective.
    assert abs(cost["Z"] - (res.fun + model["constant"])) < 1e-6 * max(1, abs(cost["Z"])), \
        "the ILP objective and the enumeration disagree"
    return {"planning": planning, **cost, "optimal": res.status == 0, "gap": res.mip_gap,
            "nodes": res.mip_node_count, "seconds": seconds}


def solve_pairs(q, s, slot_length, n_slots, c_I, c_W, time_limit, mip_rel_gap):
    """
    Same ILP when all the slots have the same length: which slot a patient gets
    no longer matters, only who shares a slot. Every planning of the slot model
    corresponds to n_slots! plannings of equal cost (permutations of the slots),
    which the branch-and-bound of solve_ilp has to rule out one by one; this
    formulation has no such symmetry.

    Variables:
        u[i] in {0, 1}    patient i is alone in a slot
        y[i, j] in {0, 1} patients i < j share a slot
    Constraints:
        u[i] + sum_j y[i, j] = 1                  every patient is alone or in exactly one pair (C1, C2)
        sum_i u[i] + sum_{i<j} y[i, j] <= n_slots at most one slot per single or pair
    Objective, with the coefficients of build_milp (L = slot_length):
        c_I * n_slots * L - c_I * sum_i a[i] u[i]
                          + sum_{i<j} (c_W w[i,j] - c_I (a[i] + a[j] + delta[i,j])) y[i, j]

    Args:
        q, s (arrays): Show probabilities and durations (minutes).
        slot_length (float): Length of every slot (minutes).
        n_slots (int): Number of slots.
        c_I, c_W (float): Costs of one minute of idle time and of waiting time.
        time_limit (float): Seconds after which HiGHS stops.
        mip_rel_gap (float): Relative gap at which HiGHS stops.

    Returns:
        dict: same keys as solve_ilp (slots numbered in the order of the pairs, then the singles).
    """
    n, L = len(q), float(slot_length)
    if n > 2 * n_slots:
        raise ValueError(f"{n} patients cannot fit in {n_slots} slots of two patients")
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a = q * np.minimum(s, L)
    c = np.zeros(n + len(pairs))
    c[:n] = -c_I * a
    for k, (i, j) in enumerate(pairs):
        w = q[i] * q[j] * min(s[i], s[j])
        delta = q[i] * q[j] * (min(s[i] + s[j], L) - min(s[i], L) - min(s[j], L))
        c[n + k] = c_W * w - c_I * (a[i] + a[j] + delta)
    A = np.zeros((n + 1, n + len(pairs)))
    A[np.arange(n), np.arange(n)] = 1                      # u[i]
    for k, (i, j) in enumerate(pairs):                      # y[i, j] counts for i and for j
        A[i, n + k] = A[j, n + k] = 1
    A[n, :] = 1                                             # number of slots used
    lower = np.r_[np.ones(n), -np.inf]
    upper = np.r_[np.ones(n), n_slots]

    start = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(A, lower, upper), bounds=Bounds(0, 1),
               integrality=np.ones(len(c)),
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False})
    seconds = time.perf_counter() - start
    if res.x is None:
        raise RuntimeError(f"HiGHS found no planning: {res.message}")
    groups = [list(pairs[k]) for k in range(len(pairs)) if res.x[n + k] > 0.5] + \
             [[i] for i in range(n) if res.x[i] > 0.5]
    planning = dict(enumerate(groups))
    cost = planning_cost(planning, q, s, [L] * n_slots, c_I, c_W)
    assert abs(cost["Z"] - (res.fun + c_I * n_slots * L)) < 1e-6 * max(1, abs(cost["Z"])), \
        "the ILP objective and the enumeration disagree"
    return {"planning": planning, **cost, "optimal": res.status == 0, "gap": res.mip_gap,
            "nodes": res.mip_node_count, "seconds": seconds}


def solve_by_enumeration(q, s, L, c_I, c_W):
    """
    Optimal planning by trying every assignment of the patients to the slots
    (T^n of them): only for small instances, to check the ILP.

    Returns:
        dict: planning and its expected cost.
    """
    n, T = len(q), len(L)
    best = None
    for slots in itertools.product(range(T), repeat=n):
        if max(np.bincount(slots, minlength=T)) > 2:
            continue
        planning = {}
        for i, t in enumerate(slots):
            planning.setdefault(t, []).append(i)
        cost = planning_cost(planning, q, s, L, c_I, c_W)
        if best is None or cost["Z"] < best["Z"] - 1e-12:
            best = {"planning": planning, **cost}
    return best


############################################################
#                                                          #
#        ------   Simple policies       -------            #
#                                                          #
############################################################

def single_booking(n, T):
    """
    One patient per slot, in the order of the requests: the first T patients
    are booked, the others are not (no overbooking).

    Returns:
        dict: {slot: [patient]}.
    """
    return {t: [t] for t in range(min(n, T))}


def greedy(q, s, L):
    """
    Greedy overbooking:
      1. the T patients with the largest expected busy time q_i * s_i each get
         a slot of their own, the longest ones in the longest slots;
      2. each remaining patient (largest q_i * s_i first) joins the
         single-patient slot with the largest expected idle time left.

    Returns:
        dict: {slot: patients}.
    """
    order = sorted(range(len(q)), key=lambda i: -q[i] * s[i])
    slots_by_length = sorted(range(len(L)), key=lambda t: -L[t])
    planning = {t: [i] for t, i in zip(slots_by_length, order)}
    for i in order[len(L):]:
        single = [t for t in planning if len(planning[t]) == 1]
        expected_idle = {t: L[t] - q[planning[t][0]] * min(s[planning[t][0]], L[t]) for t in single}
        planning[max(expected_idle, key=expected_idle.get)].append(i)
    return planning


def pairs_last(planning):
    """
    Order of the slots in the day. With slots of equal length the ILP gives the
    same cost to every order, so the order is a rule set apart: the slots with
    one patient first, then the double-booked slots, then the empty ones. The
    delay of a double-booked slot then carries over to fewer patients.

    Returns:
        dict: {slot: patients}, slots numbered in that order.
    """
    groups = [p for p in planning.values() if len(p) == 1] + [p for p in planning.values() if len(p) == 2]
    return dict(enumerate(groups))


############################################################
#                                                          #
#        ------   Simulation of the day   -------          #
#                                                          #
############################################################

def simulate_day(planning, s, L, shows, c_I, c_W, c_O):
    """
    Cost of a planning over a real day, where delays carry over: a slot starts
    at its scheduled time or when the previous patient is done, whichever is
    later. Patients of a slot are served shortest first. Idle time counts every
    minute the doctor waits for a patient, up to the end of the day; overtime
    counts every minute worked after the end of the day (the ILP of the paper
    has no overtime: it is counted here only, to judge the plannings).

    Args:
        planning (dict): {slot: patients}.
        s (array): Durations (minutes).
        L (list): Slot lengths (minutes).
        shows (array): Shape (M, n), True if patient i comes in scenario m.
            The same draws are used for every planning (common random numbers).
        c_I, c_W, c_O (float): Costs of one minute of idle time, of waiting time and of overtime.

    Returns:
        dict: Z, idle, wait and overtime for each of the M scenarios (arrays).
    """
    M = shows.shape[0]
    starts = np.concatenate([[0.0], np.cumsum(L)[:-1]])
    idle, wait, clock = np.zeros(M), np.zeros(M), np.zeros(M)
    for t in range(len(L)):
        idle += np.maximum(0.0, starts[t] - clock)
        clock = np.maximum(clock, starts[t])
        for i in sorted(planning.get(t, []), key=lambda i: s[i]):
            wait += shows[:, i] * (clock - starts[t])
            clock = clock + shows[:, i] * s[i]
    idle += np.maximum(0.0, sum(L) - clock)
    overtime = np.maximum(0.0, clock - sum(L))
    return {"Z": c_I * idle + c_W * wait + c_O * overtime, "idle": idle, "wait": wait, "overtime": overtime}
