# Quantum_MoE 
# MoE - Mixture-of-Experts
  


# Quantum_MoE_V1.py
#
# Lokalna kvantna simulacija:
# - cetiri strucnjaka sa faznim kodiranjem i refleksijom amplituda;
# - usmerivac sa dva qubita, RY i kontrolisanom RY rotacijom;
# - svaki strucnjak definise raspodelu sedam razlicitih brojeva.
#
# Klasicni deo:
# - zajednicka optimizacija analitickim gradijentima;
# - tacna normalizacija raspodele cele sedmorke;
# - deterministicka pretraga maksimuma.
#
# Ovo je mali, glatko usmeravani MoE.
# Nema nasumicnog usmeravanja, odbacivanja uzoraka ili top-k prekida.
#
# P(S | istorija) = sum_e gate_e(istorija) * P_e(S | istorija)
#
# Za strucnjaka e:
# P_e(S) = product_{i in S} w_e(i) / e_7(w_e)
#
# w_e proizlaze iz kvantnih amplituda, a e_7 je elementarni
# simetricni polinom stepena sedam.
#
# Sedam sest-qubitnih registara jednog strucnjaka pripremaju
# isto jedno-registarsko stanje. Raspodela se uslovljava
# razlicitoscu svih sedam ishoda i zanemaruje njihov redosled.
# Svaki strucnjak normalizuje se PRE mesanja.
#
# Ne alocira se statevector od 42 qubita.
# Nema projekcije amplituda u matricu 7x39.
#
# Izvor osnovnog MoE principa:
# https://arxiv.org/abs/1701.06538
#
# Tri modela za proveru:
# 1. jedan referentni strucnjak srednje vremenske skale;
# 2. cetiri strucnjaka sa naucenim stalnim udelima;
# 3. cetiri strucnjaka sa udelima zavisnim od istorije.
#
# Test ne bira hiperparametre niti menja strukturu modela.



from pathlib import Path
from itertools import combinations
from math import comb
from time import perf_counter
import random
import sys

import numpy as np

random.seed(39)
np.random.seed(39)

try:
    from scipy.optimize import minimize
    from numba import njit, prange, get_num_threads, set_num_threads
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation, DiagonalGate
    from qiskit.quantum_info import Statevector
except ImportError as exc:
    raise SystemExit(
        "Nedostaje zavisnost: "
        + str(exc)
        + ". Potrebni su NumPy, SciPy, Numba i Qiskit."
    )

set_num_threads(min(4, get_num_threads()))



FILES = (
    Path("/Users/4c/Desktop/GHQ/data/loto7_4724_k93.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4724_k93_loto_2984.csv"),
    Path("/Users/4c/Desktop/GHQ/data/loto7_4724_k93_loto_plus_1740.csv"),
)



N = 39
K = 7
EXPERTS = 4

REGULARIZATIONS = (0.001, 0.01)
GRADIENT_TOLERANCE = 2e-5
MAX_ITERATIONS = 500
MAX_RUN = 4
SEARCH_TOLERANCE = 1e-10
DETAILS = False

FAST_RATES = np.array([0.50, 0.20, 0.08, 0.03])
SLOW_RATES = FAST_RATES / 5.0

MODEL_NAMES = {
    "single": "jedan strucnjak",
    "static": "stalna mesavina",
    "dynamic": "MoE usmeravanje",
}


def load_csv(path):
    if not path.is_file():
        raise ValueError("CSV nije pronadjen: " + str(path))

    data = None
    for delimiter in (",", ";", None):
        try:
            candidate = np.loadtxt(
                path, delimiter=delimiter, dtype=np.float64, ndmin=2
            )
            if candidate.shape[1] == K:
                data = candidate
                break
        except ValueError:
            pass

    if data is None:
        raise ValueError("Potrebno je sedam numerickih kolona bez hedera.")
    if len(data) < 120:
        raise ValueError("Potrebno je najmanje 120 kola.")
    if not np.all(np.isfinite(data)):
        raise ValueError("CSV sadrzi neispravne vrednosti.")
    if not np.all(data == np.rint(data)):
        raise ValueError("CSV sadrzi brojeve koji nisu celi.")
    if np.any((data < 1) | (data > N)):
        raise ValueError("Brojevi moraju biti od 1 do 39.")

    data = np.sort(data.astype(np.int64), axis=1)
    if np.any(np.diff(data, axis=1) <= 0):
        raise ValueError("Kolo sadrzi ponovljene brojeve.")

    return data


def make_history(draws):
    """
    Features[t] koristi samo kola do t, ukljucujuci t.
    Cilj uzorka t je kolo t+1.
    Poslednji kontekst koristi celu istoriju.
    """
    total = len(draws)
    features = np.empty((total, EXPERTS, N, 4))
    contexts = np.empty((total, 4))

    fast = np.zeros((EXPERTS, N))
    slow = np.zeros((EXPERTS, N))
    previous = np.zeros(N)
    last_seen = np.full(N, -1, dtype=np.int64)

    for t, row in enumerate(draws):
        present = np.zeros(N)
        present[row - 1] = 1.0

        if t == 0:
            fast[:] = present
            slow[:] = present
        else:
            for e in range(EXPERTS):
                fast[e] += FAST_RATES[e] * (present - fast[e])
                slow[e] += SLOW_RATES[e] * (present - slow[e])

        last_seen[row - 1] = t
        known = last_seen >= 0
        waiting = np.zeros(N)
        ages = t - last_seen[known]
        waiting[known] = ages / (ages + 6.0)
        waiting[known] -= waiting[known].mean()

        for e in range(EXPERTS):
            features[t, e, :, 0] = 1.0
            features[t, e, :, 1] = present - K / N
            features[t, e, :, 2] = 4.0 * (fast[e] - slow[e])
            features[t, e, :, 3] = waiting

        contexts[t, 0] = 1.0
        contexts[t, 1] = 8.0 * np.sqrt(
            np.mean((fast[1] - slow[1]) ** 2)
        )
        contexts[t, 2] = np.dot(present, previous) / K - K / N
        contexts[t, 3] = 8.0 * np.mean(np.abs(fast[0] - fast[3]))
        previous = present

    return (
        np.ascontiguousarray(features[:-1]),
        np.ascontiguousarray(contexts[:-1]),
        np.ascontiguousarray(draws[1:] - 1),
        features[-1].copy(),
        contexts[-1].copy(),
    )


@njit
def expert_weights(theta, features):
    """
    |u> = jednaka amplituda na 39 vazecih nivoa.
    U_phi = dijagonalna fazna operacija.
    D = 2|u><u| - I.

    Posle D U_phi:
    a_i = (2*mean(exp(i*phi)) - exp(i*phi_i))/sqrt(n).

    Za normalizaciju sedmorke moze se koristiti w_i = n*|a_i|^2.
    Zajednicki faktor n se ponistava.

    phi = tanh(theta . features), pa je |phi| < 1.
    Zbog toga amplitude u ovom modelu imaju pozitivnu
    donju granicu; ne dodajemo numericki sum ni probability floor.
    """
    n = features.shape[0]
    u = np.empty(n, dtype=np.complex128)
    du = np.empty((n, 4), dtype=np.complex128)
    mean_u = 0.0j
    mean_du = np.zeros(4, dtype=np.complex128)

    for i in range(n):
        z = 0.0
        for j in range(4):
            z += theta[j] * features[i, j]
        h = np.tanh(z)
        u[i] = complex(np.cos(h), np.sin(h))
        mean_u += u[i]
        for j in range(4):
            du[i, j] = 1j * u[i] * (1.0 - h * h) * features[i, j]
            mean_du[j] += du[i, j]

    mean_u /= n
    for j in range(4):
        mean_du[j] /= n

    weights = np.empty(n)
    derivatives = np.empty((n, 4))
    for i in range(n):
        a = 2.0 * mean_u - u[i]
        weights[i] = a.real * a.real + a.imag * a.imag
        for j in range(4):
            da = 2.0 * mean_du[j] - du[i, j]
            derivatives[i, j] = 2.0 * (
                a.real * da.real + a.imag * da.imag
            )

    return weights, derivatives


@njit
def partition_and_marginals(weights, k):
    n = len(weights)
    prefix = np.zeros((n + 1, k + 1))
    suffix = np.zeros((n + 1, k + 1))
    prefix[:, 0] = 1.0
    suffix[:, 0] = 1.0

    for i in range(1, n + 1):
        for r in range(1, k + 1):
            prefix[i, r] = (
                prefix[i - 1, r]
                + weights[i - 1] * prefix[i - 1, r - 1]
            )

    for i in range(n - 1, -1, -1):
        for r in range(1, k + 1):
            suffix[i, r] = (
                suffix[i + 1, r]
                + weights[i] * suffix[i + 1, r - 1]
            )

    normalizer = prefix[n, k]
    marginals = np.empty(n)

    for i in range(n):
        excluded = 0.0
        for r in range(k):
            excluded += prefix[i, r] * suffix[i + 1, k - 1 - r]
        marginals[i] = weights[i] * excluded / normalizer

    return normalizer, marginals


@njit
def expert_logp_gradient(theta, features, target):
    weights, derivatives = expert_weights(theta, features)
    normalizer, marginals = partition_and_marginals(
        weights, len(target)
    )

    present = np.zeros(len(weights))
    logp = -np.log(normalizer)
    for value in target:
        present[value] = 1.0
        logp += np.log(weights[value])

    gradient = np.zeros(4)
    for i in range(len(weights)):
        coefficient = (present[i] - marginals[i]) / weights[i]
        for j in range(4):
            gradient[j] += coefficient * derivatives[i, j]

    return logp, gradient


@njit
def router(parameters, context):
    """
    Dva qubita:
      RY(a) na q0
      RY(b) na q1
      kontrolisana RY(c), kontrola q0, cilj q1

    Redosled strucnjaka odgovara stanjima 00, 01, 10, 11
    u Qiskit little-endian indeksiranju.
    """
    csize = len(context)
    z = np.zeros(3)

    for j in range(3):
        for m in range(csize):
            z[j] += parameters[j * csize + m] * context[m]

    h = np.tanh(z)
    a = np.pi / 2.0 + 1.2 * h[0]
    b = np.pi / 2.0 + 1.2 * h[1]
    c = 0.3 * h[2]

    t = np.cos(a / 2.0) ** 2
    u = np.cos(b / 2.0) ** 2
    v = np.cos((b + c) / 2.0) ** 2

    dt = -0.5 * np.sin(a)
    du = -0.5 * np.sin(b)
    dv = -0.5 * np.sin(b + c)

    probabilities = np.array([
        t * u,
        (1.0 - t) * v,
        t * (1.0 - u),
        (1.0 - t) * (1.0 - v),
    ])

    jacobian = np.zeros((4, 3))
    jacobian[0, 0] = dt * u
    jacobian[1, 0] = -dt * v
    jacobian[2, 0] = dt * (1.0 - u)
    jacobian[3, 0] = -dt * (1.0 - v)

    jacobian[0, 1] = t * du
    jacobian[1, 1] = (1.0 - t) * dv
    jacobian[2, 1] = -t * du
    jacobian[3, 1] = -(1.0 - t) * dv

    jacobian[1, 2] = (1.0 - t) * dv
    jacobian[3, 2] = -(1.0 - t) * dv

    scales = np.array([1.2, 1.2, 0.3])
    for j in range(3):
        factor = scales[j] * (1.0 - h[j] * h[j])
        for e in range(4):
            jacobian[e, j] *= factor

    return probabilities, jacobian


@njit(parallel=True, nogil=True)
def likelihood_kernel(parameters, features, contexts, targets):
    count = len(targets)
    experts = features.shape[1]
    dimension = len(parameters)
    blocks = 16
    losses = np.empty(count)
    partial_gradients = np.zeros((blocks, dimension))

    for block in prange(blocks):
        start = count * block // blocks
        stop = count * (block + 1) // blocks

        for t in range(start, stop):
            logps = np.empty(experts)
            expert_gradients = np.empty((experts, 4))

            for e in range(experts):
                logp, gradient = expert_logp_gradient(
                    parameters[4 * e:4 * e + 4],
                    features[t, e],
                    targets[t],
                )
                logps[e] = logp
                expert_gradients[e] = gradient

            if experts == 1:
                losses[t] = -logps[0]
                for j in range(4):
                    partial_gradients[block, j] -= expert_gradients[0, j]
            else:
                gates, gate_jacobian = router(
                    parameters[4 * experts:], contexts[t]
                )
                scores = np.empty(experts)
                for e in range(experts):
                    scores[e] = np.log(gates[e]) + logps[e]

                maximum = np.max(scores)
                total = 0.0
                for e in range(experts):
                    total += np.exp(scores[e] - maximum)
                mixture_logp = maximum + np.log(total)
                losses[t] = -mixture_logp

                responsibilities = np.exp(scores - mixture_logp)

                for e in range(experts):
                    for j in range(4):
                        partial_gradients[block, 4 * e + j] -= (
                            responsibilities[e] * expert_gradients[e, j]
                        )

                csize = contexts.shape[1]
                for j in range(3):
                    derivative = 0.0
                    for e in range(experts):
                        derivative -= (
                            responsibilities[e]
                            * gate_jacobian[e, j]
                            / gates[e]
                        )
                    for m in range(csize):
                        index = 4 * experts + j * csize + m
                        partial_gradients[block, index] += (
                            derivative * contexts[t, m]
                        )

    gradient = np.zeros(dimension)
    for block in range(blocks):
        for j in range(dimension):
            gradient[j] += partial_gradients[block, j] / count

    return losses, gradient


def model_data(features, contexts, kind):
    if kind == "single":
        return (
            np.ascontiguousarray(features[:, 1:2]),
            np.ascontiguousarray(contexts[:, :1]),
        )
    if kind == "static":
        return (
            np.ascontiguousarray(features),
            np.ascontiguousarray(contexts[:, :1]),
        )
    return np.ascontiguousarray(features), np.ascontiguousarray(contexts)


def initial_parameters(experts, context_size):
    size = 4 * experts + (3 * context_size if experts > 1 else 0)
    parameters = np.zeros(size)

    for e in range(experts):
        parameters[4 * e:4 * e + 4] = [
            0.20 + 0.07 * e,
            0.55 + 0.10 * e,
            -0.35 + 0.08 * e,
            0.20 - 0.09 * e,
        ]

    return parameters


def objective(parameters, features, contexts, targets, reg):
    losses, gradient = likelihood_kernel(
        parameters, features, contexts, targets
    )
    penalty = 0.5 * reg * np.dot(parameters, parameters) / len(parameters)
    gradient = gradient + reg * parameters / len(parameters)
    return float(losses.mean() + penalty), gradient


def fit_model(features, contexts, targets, reg, start=None):
    parameters = (
        initial_parameters(features.shape[1], contexts.shape[1])
        if start is None else np.asarray(start, dtype=np.float64).copy()
    )
    iterations = 0
    last_message = perf_counter()

    def callback(current):
        nonlocal iterations, last_message
        iterations += 1
        if perf_counter() - last_message >= 30.0:
            print("Ucenje: iteracija", iterations, flush=True)
            last_message = perf_counter()

    last_result = None

    for attempt in range(2):
        last_result = minimize(
            objective,
            parameters,
            args=(features, contexts, targets, reg),
            method="BFGS",
            jac=True,
            callback=callback,
            options={
                "gtol": GRADIENT_TOLERANCE,
                "maxiter": MAX_ITERATIONS,
            },
        )

        parameters = np.asarray(last_result.x, dtype=np.float64)
        value, gradient = objective(
            parameters, features, contexts, targets, reg
        )
        norm = float(np.max(np.abs(gradient)))

        if (
            np.isfinite(value)
            and np.all(np.isfinite(parameters))
            and np.all(np.isfinite(gradient))
            and norm <= GRADIENT_TOLERANCE
        ):
            return parameters

        if attempt == 0:
            print(
                f"Nastavak optimizacije: gradijent={norm:.6g}.",
                flush=True,
            )

    raise RuntimeError(
        f"Optimizacija nije konvergirala; gradijent={norm:.6g}; "
        + str(last_result.message)
    )


def mean_nll(parameters, features, contexts, targets):
    losses, _ = likelihood_kernel(parameters, features, contexts, targets)
    value = float(losses.mean())
    if not np.isfinite(value):
        raise RuntimeError("NLL nije konacan.")
    return value


def next_distribution(parameters, features, context):
    experts = len(features)
    weights = np.empty((experts, features.shape[1]))
    normalizers = np.empty(experts)

    for e in range(experts):
        weights[e], _ = expert_weights(
            parameters[4 * e:4 * e + 4], features[e]
        )
        normalizers[e], _ = partition_and_marginals(weights[e], K)

    gates, _ = router(parameters[4 * experts:], context)
    return weights, normalizers, gates


@njit
def product_bounds(weights, k):
    experts, n = weights.shape
    bounds = np.zeros((experts, n + 1, k + 1))
    bounds[:, :, 0] = 1.0

    for position in range(n - 1, -1, -1):
        for r in range(1, k + 1):
            for e in range(experts):
                skip = bounds[e, position + 1, r]
                take = weights[e, position] * bounds[e, position + 1, r - 1]
                bounds[e, position, r] = max(skip, take)

    return bounds


@njit
def has_five_consecutive(mask):
    return (
        mask
        & (mask >> 1)
        & (mask >> 2)
        & (mask >> 3)
        & (mask >> 4)
    ) != 0


@njit
def search_subsets(
    weights, labels, bounds, position, remaining, mask,
    mass, best_score, best_mask, visited, tolerance
):
    experts, n = weights.shape
    if n - position < remaining:
        return

    upper = 0.0
    for e in range(experts):
        upper += mass[e] * bounds[e, position, remaining]

    if best_score[0] > 0.0:
        if upper <= best_score[0] * (1.0 + tolerance):
            return

    if remaining == 0:
        visited[0] += 1
        if has_five_consecutive(mask):
            return
        probability = mass.sum()
        if probability > best_score[0]:
            best_score[0] = probability
            best_mask[0] = mask
        return

    if position >= n:
        return

    take_mass = np.empty(experts)
    take_upper = 0.0
    skip_upper = 0.0

    for e in range(experts):
        take_mass[e] = mass[e] * weights[e, position]
        take_upper += take_mass[e] * bounds[e, position + 1, remaining - 1]
        skip_upper += mass[e] * bounds[e, position + 1, remaining]

    take_mask = mask | (np.int64(1) << labels[position])

    if take_upper >= skip_upper:
        if not has_five_consecutive(take_mask):
            search_subsets(
                weights, labels, bounds, position + 1, remaining - 1,
                take_mask, take_mass, best_score, best_mask, visited, tolerance,
            )
        search_subsets(
            weights, labels, bounds, position + 1, remaining,
            mask, mass, best_score, best_mask, visited, tolerance,
        )
    else:
        search_subsets(
            weights, labels, bounds, position + 1, remaining,
            mask, mass, best_score, best_mask, visited, tolerance,
        )
        if not has_five_consecutive(take_mask):
            search_subsets(
                weights, labels, bounds, position + 1, remaining - 1,
                take_mask, take_mass, best_score, best_mask, visited, tolerance,
            )


def decode(weights, normalizers, gates, k=K, tolerance=SEARCH_TOLERANCE):
    # Redosled obilaska sluzi samo ubrzanju.
    # Gornje granice cuvaju pretragu globalnog maksimuma.
    heuristic = np.sum(gates[:, None] * weights, axis=0)
    labels = np.argsort(-heuristic, kind="stable").astype(np.int64)
    ordered_weights = np.ascontiguousarray(weights[:, labels])
    bounds = product_bounds(ordered_weights, k)

    best_score = np.array([-1.0])
    best_mask = np.zeros(1, dtype=np.int64)
    visited = np.zeros(1, dtype=np.int64)

    search_subsets(
        ordered_weights,
        labels,
        bounds,
        0,
        k,
        np.int64(0),
        np.ascontiguousarray(gates / normalizers),
        best_score,
        best_mask,
        visited,
        tolerance,
    )

    mask = int(best_mask[0])
    result = [
        i + 1 for i in range(weights.shape[1])
        if mask & (1 << i)
    ]

    if len(result) != k or best_score[0] <= 0:
        raise RuntimeError("Pretraga nije pronasla ispravan NEXT.")

    independent = np.sum(
        gates
        * np.prod(weights[:, np.array(result) - 1], axis=1)
        / normalizers
    )
    if not np.isclose(
        independent, best_score[0], rtol=1e-11, atol=1e-15
    ):
        raise RuntimeError("Provera NEXT skora nije prosla.")

    return result, float(best_score[0])


def verify_quantum_circuits():
    values = np.arange(1, N + 1, dtype=np.float64)
    features = np.column_stack((
        np.ones(N),
        np.sin(values * 0.3),
        np.cos(values * 0.2),
        np.sin(values * 0.7),
    ))
    theta = np.array([0.3, 0.6, -0.4, 0.2])
    weights, _ = expert_weights(theta, features)

    base = np.zeros(64, dtype=np.complex128)
    base[1:N + 1] = 1.0 / np.sqrt(N)
    phases = np.ones(64, dtype=np.complex128)
    phases[1:N + 1] = np.exp(1j * np.tanh(features @ theta))
    reflection = 2.0 * np.outer(base, base.conj()) - np.eye(64)

    circuit = QuantumCircuit(6)
    circuit.append(StatePreparation(base), list(range(6)))
    circuit.append(DiagonalGate(phases), list(range(6)))
    circuit.unitary(reflection, list(range(6)))

    state = Statevector.from_instruction(circuit).data
    reference = np.zeros(64)
    reference[1:N + 1] = weights / N

    if not np.allclose(
        np.abs(state) ** 2, reference, rtol=1e-10, atol=1e-11
    ):
        raise RuntimeError("Qiskit provera strucnjaka nije prosla.")

    context = np.array([1.0, 0.3, -0.2, 0.4])
    parameters = np.array([
        0.2, -0.1, 0.3, 0.15,
        -0.3, 0.2, 0.1, -0.2,
        0.1, 0.4, -0.15, 0.2,
    ])
    gates, _ = router(parameters, context)

    h = np.tanh(parameters.reshape(3, 4) @ context)
    circuit = QuantumCircuit(2)
    circuit.ry(float(np.pi / 2 + 1.2 * h[0]), 0)
    circuit.ry(float(np.pi / 2 + 1.2 * h[1]), 1)
    circuit.cry(float(0.3 * h[2]), 0, 1)

    probabilities = np.abs(Statevector.from_instruction(circuit).data) ** 2
    if not np.allclose(probabilities, gates, atol=1e-11, rtol=1e-10):
        raise RuntimeError("Qiskit provera usmerivaca nije prosla.")


def internal_checks():
    count = 6
    n = 9
    k = 3
    features = np.empty((count, 4, n, 4))
    contexts = np.empty((count, 4))

    for t in range(count):
        contexts[t] = [1.0, 0.1 * t, np.sin(t), np.cos(t)]
        for e in range(4):
            for i in range(n):
                features[t, e, i] = [
                    1.0,
                    np.sin((i + 1) * (e + 1) * 0.3 + t),
                    np.cos((i + 1) * 0.2 + t * 0.4),
                    np.sin(i * 0.7 - t * 0.3),
                ]

    targets = np.array(
        list(combinations(range(n), k))[::7][:count],
        dtype=np.int64,
    )

    for kind in ("single", "static", "dynamic"):
        x, c = model_data(features, contexts, kind)
        parameters = initial_parameters(x.shape[1], c.shape[1])
        if kind != "single":
            parameters[16:] = np.linspace(-0.2, 0.25, len(parameters) - 16)

        value, analytic = objective(parameters, x, c, targets, 0.01)
        numerical = np.empty(len(parameters))
        for j in range(len(parameters)):
            step = 1e-5
            plus = parameters.copy()
            minus = parameters.copy()
            plus[j] += step
            minus[j] -= step
            numerical[j] = (
                objective(plus, x, c, targets, 0.01)[0]
                - objective(minus, x, c, targets, 0.01)[0]
            ) / (2.0 * step)

        error = np.max(
            np.abs(analytic - numerical)
            / np.maximum(1.0, np.abs(numerical))
        )
        if not np.isfinite(value) or error > 2e-5:
            raise RuntimeError("Analiticki gradijent nije prosao proveru.")

        permutation = np.array([4, 0, 7, 2, 8, 1, 6, 3, 5])
        inverse = np.argsort(permutation)
        permuted_value = objective(
            parameters,
            np.ascontiguousarray(x[:, :, permutation]),
            c,
            np.ascontiguousarray(inverse[targets]),
            0.01,
        )[0]
        if not np.isclose(value, permuted_value, atol=1e-10):
            raise RuntimeError("Model zavisi od proizvoljnog redosleda oznaka.")

    parameters = initial_parameters(4, 4)
    weights = np.empty((4, n))
    for e in range(4):
        weights[e], _ = expert_weights(
            parameters[4 * e:4 * e + 4], features[0, e]
        )
    gates, _ = router(parameters[16:], contexts[0])

    for test_k in (3, 5):
        normalizers = np.array([
            partition_and_marginals(w, test_k)[0] for w in weights
        ])
        total = 0.0
        brute_best = -1.0

        for combination in combinations(range(n), test_k):
            probability = float(np.sum(
                gates * np.prod(weights[:, combination], axis=1)
                / normalizers
            ))
            total += probability
            mask = sum(1 << i for i in combination)
            if not has_five_consecutive(np.int64(mask)):
                brute_best = max(brute_best, probability)

        if not np.isclose(total, 1.0, atol=1e-11):
            raise RuntimeError("Mesavina nije normalizovana.")

        _, decoded = decode(
            weights, normalizers, gates, k=test_k, tolerance=0.0
        )
        if not np.isclose(decoded, brute_best, rtol=1e-11, atol=1e-13):
            raise RuntimeError("Dekoder se ne poklapa sa iscrpnom proverom.")

    uniform_z, uniform_marginals = partition_and_marginals(np.ones(N), K)
    if not np.isclose(uniform_z, comb(N, K)):
        raise RuntimeError("Provera normalizatora nije prosla.")
    if not np.allclose(uniform_marginals, K / N):
        raise RuntimeError("Provera marginala nije prosla.")

    synthetic = np.array([
        sorted({((5 * t + 3 * j) % N) + 1 for j in range(K)})
        for t in range(20)
    ], dtype=np.int64)
    full = make_history(synthetic)
    prefix = make_history(synthetic[:12])
    if not np.allclose(full[0][:11], prefix[0]):
        raise RuntimeError("Buduci podaci menjaju ranije ulaze.")
    if not np.allclose(full[1][:11], prefix[1]):
        raise RuntimeError("Buduci podaci menjaju raniji kontekst.")

    verify_quantum_circuits()


def process_csv(path):
    print("CSV:", path.name, flush=True)
    draws = load_csv(path)
    features, contexts, targets, next_features, next_context = make_history(draws)

    count = len(targets)
    train_end = int(0.70 * count)
    validation_end = int(0.85 * count)

    selected = {}
    test_scores = {}

    for kind in ("single", "static", "dynamic"):
        x, c = model_data(features, contexts, kind)
        best_validation = np.inf
        best_parameters = None
        best_reg = None

        for reg in REGULARIZATIONS:
            print(
                f"Ucenje: {MODEL_NAMES[kind]} | reg={reg:g}.",
                flush=True,
            )
            try:
                parameters = fit_model(
                    x[:train_end],
                    c[:train_end],
                    targets[:train_end],
                    reg,
                )
                validation = mean_nll(
                    parameters,
                    x[train_end:validation_end],
                    c[train_end:validation_end],
                    targets[train_end:validation_end],
                )
                if validation < best_validation:
                    best_validation = validation
                    best_parameters = parameters
                    best_reg = reg
            except RuntimeError as exc:
                print(f"PROVERA: kandidat odbacen: {exc}")

        if best_parameters is None:
            raise RuntimeError(
                "Nijedan kandidat nije prosao: " + MODEL_NAMES[kind]
            )

        print(
            f"Validacija NLL: {best_validation:.6f} "
            f"| {MODEL_NAMES[kind]} | reg={best_reg:g}"
        )
        print("Ucenje za test:", MODEL_NAMES[kind], flush=True)

        test_parameters = fit_model(
            x[:validation_end],
            c[:validation_end],
            targets[:validation_end],
            best_reg,
            start=best_parameters,
        )
        test_nll = mean_nll(
            test_parameters,
            x[validation_end:],
            c[validation_end:],
            targets[validation_end:],
        )
        print(f"Test NLL: {test_nll:.6f} | {MODEL_NAMES[kind]}")

        selected[kind] = (best_reg, test_parameters)
        test_scores[kind] = test_nll

        if kind == "dynamic":
            test_gates = np.array([
                router(test_parameters[16:], context)[0]
                for context in c[validation_end:]
            ])
            gate_variation = float(np.max(np.std(test_gates, axis=0)))
            print(
                "Promenljivost udela usmerivaca na testu: "
                f"{gate_variation:.6f}."
            )

    gain_single = test_scores["single"] - test_scores["dynamic"]
    gain_static = test_scores["static"] - test_scores["dynamic"]
    print(
        "Test dobitak MoE, pozitivan je bolji: "
        f"prema jednom={gain_single:.6f} "
        f"| prema stalnoj mesavini={gain_static:.6f}."
    )

    # Test poredi modele; ne prebacuje zavrsni model na osnovu testa.
    reg, start = selected["dynamic"]
    print("Zavrsno MoE ucenje: svi istorijski prelazi.", flush=True)

    parameters = fit_model(
        features, contexts, targets, reg, start=start
    )
    weights, normalizers, gates = next_distribution(
        parameters, next_features, next_context
    )

    print("Udeli strucnjaka:", [round(float(v), 4) for v in gates])
    print("NEXT: deterministicka pretraga zajednicke raspodele.", flush=True)
    prediction, probability = decode(weights, normalizers, gates)
    print("NEXT:", prediction)

    if DETAILS:
        print(f"Verovatnoca cele kombinacije: {probability:.12g}")

    return {
        "name": path.name,
        "draws": draws,
        "next": tuple(prediction),
        "parameters": parameters,
        "weights": weights,
        "gates": gates,
    }


def main():
    print("Quantum MoE V1: cetiri strucnjaka i nauceno usmeravanje.")
    print("Fazna interferencija; usmerivac sa dva qubita.")
    print("Lokalna simulacija; klasicna optimizacija i tacan NLL.")
    print("NEXT: najvise cetiri uzastopna broja.")
    print("Pokretanje internih provera.", flush=True)

    internal_checks()
    print("Interne provere: prosle.", flush=True)

    results = []
    failures = 0

    for path in FILES:
        print()
        try:
            results.append(process_csv(path))
        except KeyboardInterrupt:
            print()
            print("Obrada prekinuta korisnickim zahtevom.")
            return 130
        except Exception as exc:
            failures += 1
            print("STOP za ovaj CSV:", str(exc))

    for i in range(len(results)):
        for j in range(i):
            a = results[j]
            b = results[i]
            if a["next"] != b["next"]:
                continue

            same_data = (
                a["draws"].shape == b["draws"].shape
                and np.array_equal(a["draws"], b["draws"])
            )
            print()
            print("PROVERA: isti NEXT za", a["name"], "i", b["name"])
            print(
                f"Isti CSV sadrzaj: {same_data} "
                f"| razlika parametara: "
                f"{np.linalg.norm(a['parameters'] - b['parameters']):.6g} "
                f"| razlika tezina: "
                f"{np.linalg.norm(a['weights'] - b['weights']):.6g} "
                f"| razlika usmeravanja: "
                f"{np.linalg.norm(a['gates'] - b['gates']):.6g}"
            )

    if failures:
        print()
        print("Obrada nije kompletna: neki CSV-ovi nisu prosli provere.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("Obrada prekinuta korisnickim zahtevom.")
        sys.exit(130)
    except Exception as exc:
        print("STOP:", str(exc))
        sys.exit(1)



"""
Quantum MoE V1: cetiri strucnjaka i nauceno usmeravanje.
Fazna interferencija; usmerivac sa dva qubita.
Lokalna simulacija; klasicna optimizacija i tacan NLL.
NEXT: najvise cetiri uzastopna broja.
Pokretanje internih provera.
Interne provere: prosle.



CSV: loto7_4724_k93.csv
Ucenje: jedan strucnjak | reg=0.001.
Ucenje: jedan strucnjak | reg=0.01.
Validacija NLL: 16.549209 | jedan strucnjak | reg=0.01
Ucenje za test: jedan strucnjak
Test NLL: 16.547718 | jedan strucnjak
Ucenje: stalna mesavina | reg=0.001.
Ucenje: stalna mesavina | reg=0.01.
Validacija NLL: 16.549266 | stalna mesavina | reg=0.01
Ucenje za test: stalna mesavina
Test NLL: 16.546997 | stalna mesavina
Ucenje: MoE usmeravanje | reg=0.001.
Ucenje: MoE usmeravanje | reg=0.01.
Validacija NLL: 16.549315 | MoE usmeravanje | reg=0.01
Ucenje za test: MoE usmeravanje
Test NLL: 16.546905 | MoE usmeravanje
Promenljivost udela usmerivaca na testu: 0.012641.
Test dobitak MoE, pozitivan je bolji: prema jednom=0.000812 | prema stalnoj mesavini=0.000092.
Zavrsno MoE ucenje: svi istorijski prelazi.
Udeli strucnjaka: [0.1832, 0.1691, 0.3574, 0.2903]
NEXT: deterministicka pretraga zajednicke raspodele.
NEXT: [5, 8, 10, 21, 22, 26, 36]



CSV: loto7_4724_k93_loto_2984.csv
Ucenje: jedan strucnjak | reg=0.001.
Ucenje: jedan strucnjak | reg=0.01.
Validacija NLL: 16.550239 | jedan strucnjak | reg=0.01
Ucenje za test: jedan strucnjak
Test NLL: 16.546780 | jedan strucnjak
Ucenje: stalna mesavina | reg=0.001.
Ucenje: stalna mesavina | reg=0.01.
Validacija NLL: 16.549717 | stalna mesavina | reg=0.01
Ucenje za test: stalna mesavina
Test NLL: 16.546946 | stalna mesavina
Ucenje: MoE usmeravanje | reg=0.001.
Ucenje: MoE usmeravanje | reg=0.01.
Validacija NLL: 16.549262 | MoE usmeravanje | reg=0.01
Ucenje za test: MoE usmeravanje
Test NLL: 16.548401 | MoE usmeravanje
Promenljivost udela usmerivaca na testu: 0.080926.
Test dobitak MoE, pozitivan je bolji: prema jednom=-0.001621 | prema stalnoj mesavini=-0.001455.
Zavrsno MoE ucenje: svi istorijski prelazi.
Udeli strucnjaka: [0.4105, 0.2019, 0.2558, 0.1318]
NEXT: deterministicka pretraga zajednicke raspodele.
NEXT: [2, 9, 13, 16, 18, 24, 25]



CSV: loto7_4724_k93_loto_plus_1740.csv
Ucenje: jedan strucnjak | reg=0.001.
Ucenje: jedan strucnjak | reg=0.01.
Validacija NLL: 16.547238 | jedan strucnjak | reg=0.001
Ucenje za test: jedan strucnjak
Test NLL: 16.544157 | jedan strucnjak
Ucenje: stalna mesavina | reg=0.001.
Ucenje: stalna mesavina | reg=0.01.
Validacija NLL: 16.543952 | stalna mesavina | reg=0.001
Ucenje za test: stalna mesavina
Test NLL: 16.547520 | stalna mesavina
Ucenje: MoE usmeravanje | reg=0.001.
Ucenje: MoE usmeravanje | reg=0.01.
Validacija NLL: 16.547514 | MoE usmeravanje | reg=0.001
Ucenje za test: MoE usmeravanje
Test NLL: 16.550815 | MoE usmeravanje
Promenljivost udela usmerivaca na testu: 0.235405.
Test dobitak MoE, pozitivan je bolji: prema jednom=-0.006657 | prema stalnoj mesavini=-0.003295.
Zavrsno MoE ucenje: svi istorijski prelazi.
Udeli strucnjaka: [0.2417, 0.7312, 0.0139, 0.0132]
NEXT: deterministicka pretraga zajednicke raspodele.
NEXT: [21, 24, 25, 33, 35, 37, 38]
"""



"""
MoE mehanizam: više stručnjaka uči različite obrasce, a naučeni usmerivač određuje njihov doprinos za svaki ulaz. 

- Nekoliko malih stručnjaka za raspodelu cele sedmorke, sa mogućnošću da nauče različite vremenske zavisnosti.
- Usmerivač koji koristi samo prethodna kola i određuje kome dati veći doprinos pri sledećoj predikciji.
- Zajedničko učenje stručnjaka i usmerivača, pa izbor NEXT-a iz njihove objedinjene raspodele — bez glasanja za pojedinačne brojeve.

Ako se različiti obrasci pojavljuju u različitim delovima istorije, model može naučiti kada je koji stručnjak koristan. 
Stalni prosek modela to ne može da prilagodi svakom ulazu.

Glavni rizici su preučenje i svođenje svih odluka na jednog stručnjaka. 
Stabilnost učenja je poznat problem MoE arhitektura. ST-MoE

Ključna provera je poređenje sa jednim stručnjakom i stalnom mešavinom, da utvrdim doprinos naučenog usmeravanja.

Naučeno usmeravanje omogućava modelu da prema prethodnim kolima prilagođava doprinos različitih stručnjaka.

Četiri kvantna stručnjaka i usmerivač koji iz prethodnih kola uči njihov doprinos. 
Poređenje sa jednim stručnjakom i stalnom mešavinom, uz odvojenu obradu sva tri CSV-a.
Četiri kvantna stručnjaka zajednički uče sa usmerivačem. 
V1 koristi glatku mešavinu njihovih raspodela, što je adaptacija MoE principa. 

Potrebni su NumPy, SciPy, Numba i Qiskit. 
"""
