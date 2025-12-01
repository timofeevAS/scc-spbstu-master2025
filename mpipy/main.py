from mpi4py import MPI
import sys
import math
import time
import random

BOTTOM_TEMPERATURE_BOUND = 1e-6
LOG_EVERY_ITER = 1000  # как в C-версии


# ================== QAP structures ================== #

class QAPProblem:
    def __init__(self, n, flow, distance):
        self.n = n
        self.flow = flow      # flat list length n*n
        self.distance = distance  # flat list length n*n


class QAPSolution:
    def __init__(self, n):
        self.n = n
        self.perm = list(range(n))  # permutation
        self.cost = 0.0


def idx(n, i, j):
    return i * n + j


# ================== QAP operations ================== #

def qap_eval(problem: QAPProblem, sol: QAPSolution) -> float:
    n = problem.n
    F = problem.flow
    D = problem.distance
    p = sol.perm

    cost = 0.0
    for i in range(n):
        li = p[i]
        for j in range(n):
            lj = p[j]
            cost += F[idx(n, i, j)] * D[idx(n, li, lj)]
    return cost


def qap_delta_swap(problem: QAPProblem, sol: QAPSolution, i: int, j: int) -> float:
    if i == j:
        return 0.0

    if i > j:
        i, j = j, i

    n = problem.n
    F = problem.flow
    D = problem.distance
    p = sol.perm

    pi = p[i]
    pj = p[j]

    delta = 0.0

    for k in range(n):
        if k == i or k == j:
            continue
        pk = p[k]

        delta += (F[idx(n, i, k)] - F[idx(n, j, k)]) * (D[idx(n, pj, pk)] - D[idx(n, pi, pk)]) \
               + (F[idx(n, k, i)] - F[idx(n, k, j)]) * (D[idx(n, pk, pj)] - D[idx(n, pk, pi)])

    delta += (F[idx(n, i, i)] - F[idx(n, j, j)]) * (D[idx(n, pj, pj)] - D[idx(n, pi, pi)]) \
           + (F[idx(n, i, j)] - F[idx(n, j, i)]) * (D[idx(n, pj, pi)] - D[idx(n, pi, pj)])

    return delta


def qap_apply_swap(sol: QAPSolution, i: int, j: int, delta: float):
    sol.perm[i], sol.perm[j] = sol.perm[j], sol.perm[i]
    sol.cost += delta


# ================== Solution helpers ================== #

def sol_identity(sol: QAPSolution):
    for i in range(sol.n):
        sol.perm[i] = i


def sol_shuffle(sol: QAPSolution, rng: random.Random):
    # Fisher–Yates
    n = sol.n
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        sol.perm[i], sol.perm[j] = sol.perm[j], sol.perm[i]


def sol_copy(src: QAPSolution, dst: QAPSolution):
    if dst.n != src.n:
        dst.n = src.n
        dst.perm = [0] * src.n
    dst.perm[:] = src.perm
    dst.cost = src.cost


# ================== Load QAP instance ================== #

def load_qap(path: str) -> QAPProblem:
    with open(path, 'r') as f:
        # n may be on first line
        tokens = []
        for line in f:
            tokens.extend(line.split())
        if not tokens:
            raise ValueError("Empty QAP file")

    # first token is n
    n = int(tokens[0])
    expected = 1 + n * n + n * n
    if len(tokens) < expected:
        raise ValueError(f"Not enough numbers in QAP file: expected {expected}, got {len(tokens)}")

    # next n^2 flow, then n^2 distance
    flow_tokens = tokens[1:1 + n * n]
    dist_tokens = tokens[1 + n * n:1 + 2 * n * n]

    flow = [float(x) for x in flow_tokens]
    dist = [float(x) for x in dist_tokens]

    return QAPProblem(n, flow, dist)


# ================== Simulated Annealing (per rank) ================== #

def sa_run(problem: QAPProblem, iters: int, random_start: int, rank: int, nprocs: int):
    """
    Один прогон SA, полностью локальный для MPI-ранка.
    Возвращает:
      best_cost, best_perm, trace_time[], trace_cost[]
    """
    n = problem.n

    # RNG: один seed, но разный offset через rank
    base_seed = 12345
    rng = random.Random(base_seed + rank * 1000003)

    current = QAPSolution(n)
    best = QAPSolution(n)

    sol_identity(current)
    if random_start:
        sol_shuffle(current, rng)

    current.cost = qap_eval(problem, current)
    sol_copy(current, best)

    T0 = 300000.0
    alpha = 0.9992

    T = T0

    # логирование
    max_logs = iters // LOG_EVERY_ITER + 5
    trace_cost = []
    trace_time = []

    t_start = MPI.Wtime()

    for it in range(iters):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue

        delta = qap_delta_swap(problem, current, i, j)

        accept = False
        if delta < 0.0:
            accept = True
        else:
            u = rng.random()
            prob = math.exp(-delta / T) if T > 0.0 else 0.0
            if u < prob:
                accept = True

        if accept:
            qap_apply_swap(current, i, j, delta)
            if current.cost < best.cost:
                sol_copy(current, best)

        T *= alpha
        if T < BOTTOM_TEMPERATURE_BOUND:
            T = BOTTOM_TEMPERATURE_BOUND

        if it % LOG_EVERY_ITER == 0:
            elapsed = MPI.Wtime() - t_start
            trace_time.append(elapsed)
            trace_cost.append(best.cost)

    return best.cost, best.perm, trace_time, trace_cost


# ================== MAIN (MPI) ================== #

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: mpirun -np N python sa_mpi.py instance.dat [iters] [random_start]")
        MPI.Finalize()
        return

    path = sys.argv[1]
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    random_start = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    if rank == 0:
        print(f"[MPI SA] processes = {nprocs}")
        print(f"instance   = {path}")
        print(f"iters      = {iters}")
        print(f"random_start = {random_start}")

    # Каждый процесс сам читает файл (для простоты)
    problem = load_qap(path)

    # Локальный SA на ранке
    best_cost, best_perm, trace_time, trace_cost = sa_run(
        problem, iters, random_start, rank, nprocs
    )

    # Лог в CSV по ранку
    fname = f"trace_rank_{rank}.csv"
    with open(fname, 'w') as f:
        f.write("time,best_cost\n")
        for t, c in zip(trace_time, trace_cost):
            f.write(f"{t:.6f},{c:.10f}\n")

    # Собираем best_cost на rank 0
    all_costs = comm.gather(best_cost, root=0)

    if rank == 0:
        global_best = all_costs[0]
        best_rank = 0
        for r, c in enumerate(all_costs):
            if c < global_best:
                global_best = c
                best_rank = r

        print(f"\nGLOBAL BEST COST = {global_best:.10f} (from rank {best_rank})")

    else:
        best_rank = None

    # Рассылаем best_rank всем
    best_rank = comm.bcast(best_rank, root=0)

    # Передаём лучшую пермутацию от best_rank к root
    n = problem.n
    if rank == best_rank:
        # отправляем как список int
        comm.send(best_perm, dest=0, tag=42)

    if rank == 0:
        perm = comm.recv(source=best_rank, tag=42)
        print("Permutation:")
        print(" ".join(str(x) for x in perm))


if __name__ == "__main__":
    main()
