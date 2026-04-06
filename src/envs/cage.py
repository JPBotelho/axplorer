import numpy as np
from math import comb
from numba import njit

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag


# ────────────────────────────────────────────────────────────
# Numba-accelerated kernels (operate on raw uint8 adj matrix)
# ────────────────────────────────────────────────────────────

@njit(cache=True)
def _has_short_cycle(A, N, u, v):
    """Return True if adding edge (u,v) would create a triangle or 4-cycle."""
    # Triangle: u and v share a neighbor
    for w in range(N):
        if A[u, w] == 1 and A[v, w] == 1:
            return True
    # 4-cycle: a neighbor of u and a neighbor of v are connected
    for a in range(N):
        if A[u, a] == 0 or a == v:
            continue
        for b in range(N):
            if A[v, b] == 0 or b == u or b == a:
                continue
            if A[a, b] == 1:
                return True
    return False


@njit(cache=True)
def _count_short_cycles(A, N, u, v):
    """Count triangles + 4-cycles involving edge (u,v)."""
    count = 0
    # Triangles: common neighbors
    for w in range(N):
        if A[u, w] == 1 and A[v, w] == 1:
            count += 1
    # 4-cycles through (u,v): u-v-a-b-u
    for a in range(N):
        if A[v, a] == 0 or a == u:
            continue
        for b in range(N):
            if A[u, b] == 0 or b == v or b == a:
                continue
            if A[a, b] == 1:
                count += 1
    return count


@njit(cache=True)
def _greedy_construct_numba(N, k, seed):
    """Build a random graph greedily: add edges avoiding 3/4-cycles, respecting degree k."""
    np.random.seed(seed)
    A = np.zeros((N, N), dtype=np.uint8)
    degrees = np.zeros(N, dtype=np.int32)

    # Build shuffled candidate list
    num_pairs = N * (N - 1) // 2
    candidates = np.empty((num_pairs, 2), dtype=np.int32)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            candidates[idx, 0] = i
            candidates[idx, 1] = j
            idx += 1

    # Fisher-Yates shuffle
    for i in range(num_pairs - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        candidates[i, 0], candidates[j, 0] = candidates[j, 0], candidates[i, 0]
        candidates[i, 1], candidates[j, 1] = candidates[j, 1], candidates[i, 1]

    for idx in range(num_pairs):
        i = candidates[idx, 0]
        j = candidates[idx, 1]
        if degrees[i] >= k or degrees[j] >= k:
            continue
        if not _has_short_cycle(A, N, i, j):
            A[i, j] = 1
            A[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1

    return A


@njit(cache=True)
def _greedy_add_edges_numba(A, N, k, seed):
    """Greedily add edges respecting degree k and girth >= 5."""
    np.random.seed(seed)
    degrees = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            degrees[i] += A[i, j]

    # Collect candidates
    num_pairs = N * (N - 1) // 2
    candidates = np.empty((num_pairs, 2), dtype=np.int32)
    count = 0
    for i in range(N):
        if degrees[i] >= k:
            continue
        for j in range(i + 1, N):
            if degrees[j] >= k:
                continue
            if A[i, j] == 0:
                candidates[count, 0] = i
                candidates[count, 1] = j
                count += 1

    # Shuffle
    for i in range(count - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        candidates[i, 0], candidates[j, 0] = candidates[j, 0], candidates[i, 0]
        candidates[i, 1], candidates[j, 1] = candidates[j, 1], candidates[i, 1]

    for idx in range(count):
        i = candidates[idx, 0]
        j = candidates[idx, 1]
        if degrees[i] >= k or degrees[j] >= k:
            continue
        if not _has_short_cycle(A, N, i, j):
            A[i, j] = 1
            A[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1

    return A


@njit(cache=True)
def _remove_short_cycle_edges(A, N, max_iters):
    """Remove edges involved in short cycles, highest count first. Returns cleaned A."""
    for _ in range(max_iters):
        # Quick check: any triangles?
        has_bad = False
        for i in range(N):
            if has_bad:
                break
            for j in range(i + 1, N):
                if A[i, j] == 0:
                    continue
                for w in range(N):
                    if A[i, w] == 1 and A[j, w] == 1:
                        has_bad = True
                        break
                if has_bad:
                    break

        if not has_bad:
            # Check 4-cycles: any pair with >=2 common neighbors?
            for i in range(N):
                if has_bad:
                    break
                for j in range(i + 1, N):
                    cn = 0
                    for w in range(N):
                        if A[i, w] == 1 and A[j, w] == 1:
                            cn += 1
                    if cn >= 2:
                        has_bad = True
                        break

        if not has_bad:
            break

        # Find worst edge
        worst_i = -1
        worst_j = -1
        worst_count = -1
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] == 0:
                    continue
                c = _count_short_cycles(A, N, i, j)
                if c > worst_count:
                    worst_count = c
                    worst_i = i
                    worst_j = j

        if worst_count <= 0:
            break

        A[worst_i, worst_j] = 0
        A[worst_j, worst_i] = 0

    return A


@njit(cache=True)
def _fix_overdegree(A, N, k):
    """Remove edges from over-degree vertices."""
    degrees = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for j in range(N):
            degrees[i] += A[i, j]

    for v in range(N):
        while degrees[v] > k:
            # Find neighbor with highest degree
            best_n = -1
            best_d = -1
            for w in range(N):
                if A[v, w] == 1 and degrees[w] > best_d:
                    best_d = degrees[w]
                    best_n = w
            if best_n < 0:
                break
            A[v, best_n] = 0
            A[best_n, v] = 0
            degrees[v] -= 1
            degrees[best_n] -= 1

    return A


@njit(cache=True)
def _score_violations(A, N, k):
    """Fast violation count (lower = better). 0 = perfect."""
    deg_pen = 0
    for i in range(N):
        d = 0
        for j in range(N):
            d += A[i, j]
        diff = d - k
        if diff < 0:
            deg_pen -= diff
        else:
            deg_pen += diff

    triangles = 0
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j] == 0:
                continue
            for w in range(j + 1, N):
                if A[i, w] == 1 and A[j, w] == 1:
                    triangles += 1

    four_cycles = 0
    for i in range(N):
        for j in range(i + 1, N):
            cn = 0
            for w in range(N):
                if A[i, w] == 1 and A[j, w] == 1:
                    cn += 1
            if cn >= 2:
                four_cycles += cn * (cn - 1) // 2
    four_cycles //= 2

    return deg_pen + triangles + four_cycles


@njit(cache=True)
def _deep_search(A, N, k, max_rounds, seed):
    """Stochastic local search: repeatedly perturb and re-add, keeping improvements.
    Returns array of all snapshots where score improved (including final).
    Each snapshot is a flattened N*N adjacency matrix.
    Also returns an array of (round_number, violation_count) for each improvement.
    """
    np.random.seed(seed)

    # Pre-allocate space for improved snapshots (at most max_rounds, but typically few)
    max_snapshots = 1000
    snapshots = np.zeros((max_snapshots, N, N), dtype=np.uint8)
    snapshot_info = np.zeros((max_snapshots, 2), dtype=np.int64)  # (round, violations)
    n_snapshots = 0

    best_A = A.copy()
    best_viol = _score_violations(A, N, k)

    # Save initial state as first snapshot
    if n_snapshots < max_snapshots:
        snapshots[n_snapshots] = best_A
        snapshot_info[n_snapshots, 0] = 0
        snapshot_info[n_snapshots, 1] = best_viol
        n_snapshots += 1

    for rnd in range(max_rounds):
        # Save current state
        saved_A = A.copy()

        # Collect existing edges
        edges_i = np.empty(N * k, dtype=np.int32)
        edges_j = np.empty(N * k, dtype=np.int32)
        n_edges = 0
        for i in range(N):
            for j in range(i + 1, N):
                if A[i, j] == 1:
                    edges_i[n_edges] = i
                    edges_j[n_edges] = j
                    n_edges += 1

        if n_edges == 0:
            break

        # Remove 2-4 random edges
        n_remove = min(2 + np.random.randint(0, 3), n_edges)
        removed = np.zeros(n_edges, dtype=np.uint8)
        for _ in range(n_remove):
            idx = np.random.randint(0, n_edges)
            while removed[idx] == 1:
                idx = np.random.randint(0, n_edges)
            removed[idx] = 1
            ei, ej = edges_i[idx], edges_j[idx]
            A[ei, ej] = 0
            A[ej, ei] = 0

        # Greedily re-add edges (different random order → different result)
        A = _greedy_add_edges_numba(A, N, k, np.random.randint(0, 2**31))

        viol = _score_violations(A, N, k)
        if viol < best_viol:
            best_viol = viol
            best_A = A.copy()
            # Save improved snapshot
            if n_snapshots < max_snapshots:
                snapshots[n_snapshots] = best_A
                snapshot_info[n_snapshots, 0] = rnd + 1
                snapshot_info[n_snapshots, 1] = best_viol
                n_snapshots += 1
            if best_viol == 0:
                break
        else:
            # Revert
            A = saved_A

    return snapshots[:n_snapshots], snapshot_info[:n_snapshots], n_snapshots


# ────────────────────────────────────────────────────────────
# DataPoint and Environment classes
# ────────────────────────────────────────────────────────────

class CageDataPoint(DataPoint):
    MAKE_OBJECT_CANONICAL = False
    K_REG = 8  # target degree regularity

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((self.N, self.N), dtype=np.uint8)

        if init:
            self.data = _greedy_construct_numba(N, self.K_REG, np.random.randint(0, 2**31))
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score()

    def calc_score(self):
        k = self.K_REG
        A = self.data.astype(np.int32)
        N = self.N

        # Degree penalty
        degrees = A.sum(axis=1)
        degree_penalty = int(np.sum(np.abs(degrees - k)))

        # Triangles: diagonal of A^3 via A2 * A element-wise
        A2 = A @ A
        triangles = int(np.sum(np.sum(A2 * A, axis=1))) // 6

        # 4-cycles: for each pair (i<j), C(A2[i,j], 2), divided by 2
        four_cycles = 0
        for i in range(N):
            for j in range(i + 1, N):
                cn = A2[i, j]
                if cn >= 2:
                    four_cycles += cn * (cn - 1) // 2
        four_cycles //= 2

        max_triangles = comb(N, 3)
        max_four_cycles = 3 * comb(N, 4)
        max_degree = N * k

        self.score = (max_triangles - triangles) + (max_four_cycles - four_cycles) + (max_degree - degree_penalty)

    def target_score(self):
        """The score of a perfect (k-regular, girth>=5) graph on N vertices."""
        return comb(self.N, 3) + 3 * comb(self.N, 4) + self.N * self.K_REG

    def calc_features(self):
        idx = np.triu_indices(self.N, k=1)
        self.features = ",".join(map(str, self.data[idx]))

    def local_search(self, improve_with_local_search):
        """Remove edges causing short cycles, fix overdegree, then greedily re-add."""
        N = self.N
        k = self.K_REG

        self.data = _remove_short_cycle_edges(self.data, N, 500)
        self.data = _fix_overdegree(self.data, N, k)

        if improve_with_local_search:
            self.data = _greedy_add_edges_numba(self.data, N, k, np.random.randint(0, 2**31))

        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

    def deep_local_search(self, max_rounds=10000):
        """Deep stochastic local search: perturb edges and greedily re-add, keeping improvements.
        Returns a list of all intermediate improved graphs (as CageDataPoints)."""
        N = self.N
        k = self.K_REG
        # First clean up
        self.data = _remove_short_cycle_edges(self.data, N, 500)
        self.data = _fix_overdegree(self.data, N, k)
        self.data = _greedy_add_edges_numba(self.data, N, k, np.random.randint(0, 2**31))
        # Then deep search
        snapshots, snapshot_info, n_snapshots = _deep_search(
            self.data, N, k, max_rounds, np.random.randint(0, 2**31)
        )

        # Build list of all improved graphs
        improved = []
        for i in range(n_snapshots):
            dp = CageDataPoint(N=N, init=False)
            dp.origin = getattr(self, "origin", "unknown")
            dp.data = snapshots[i].copy()
            dp.calc_features()
            dp.calc_score()
            improved.append(dp)

        # Set self to the best (last snapshot)
        if n_snapshots > 0:
            self.data = snapshots[n_snapshots - 1].copy()
        if self.MAKE_OBJECT_CANONICAL:
            self.data = sort_graph_based_on_degree(self.data)
        self.calc_features()
        self.calc_score()

        return improved, snapshot_info[:n_snapshots]

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL, cls.K_REG = pars

    @classmethod
    def _save_class_params(cls):
        return (cls.MAKE_OBJECT_CANONICAL, cls.K_REG)


class CageEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = CageDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.K_REG = params.k
        encoding_augmentation = random_symmetry_adj_matrix if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric, self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric, self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation
            )
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(
                self.data_class,
                params.N,
                self.k,
                self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS,
                pow2base=params.pow2base,
                encoding_augmentation=encoding_augmentation,
            )
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=79, help="Number of vertices in the cage graph")
        parser.add_argument("--k", type=int, default=8, help="Target degree regularity for the cage")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer", help="single_integer/sequence_k_tokens/adjacency")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false", help="sort the graph node names based on its indegree")
        parser.add_argument(
            "--augment_data_representation", type=bool_flag, default="false", help="augment the data representation with predefined function"
        )
        parser.add_argument("--pow2base", type=int, default=1, help="Number of adjacency entries to code together")
