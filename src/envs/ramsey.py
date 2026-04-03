"""
Ramsey graph environment for Axplorer.

score(G) = C(N,s) + C(N,t) - (count_Ks_in_G + count_Kt_in_complement_G)

Score is always >= 0. Max score = C(N,s) + C(N,t) iff G is a valid (s,t,N)-Ramsey graph.
Local search: greedy hill climbing + simulated annealing.
"""

import math
import time
import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix
from src.utils import bool_flag

try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

if _NUMBA:
    @_njit(cache=True)
    def _nb_popcount(x):
        count = np.int64(0)
        while x != np.int64(0):
            x &= x - np.int64(1)
            count += np.int64(1)
        return count

    @_njit(cache=True)
    def _nb_ctz(x):
        count = np.int64(0)
        while (x & np.int64(1)) == np.int64(0):
            x >>= np.int64(1)
            count += np.int64(1)
        return count

    @_njit(cache=True)
    def _nb_clique_rec(adj, candidates_mask, depth, s):
        if depth == s:
            return np.int64(1)
        total = np.int64(0)
        m = candidates_mask
        while m != np.int64(0):
            lsb = m & (-m)
            v = _nb_ctz(lsb)
            m ^= lsb
            higher_mask = ~((np.int64(1) << (v + np.int64(1))) - np.int64(1))
            higher_nbrs = adj[v] & candidates_mask & higher_mask
            if _nb_popcount(higher_nbrs) >= s - depth - np.int64(1):
                total += _nb_clique_rec(adj, higher_nbrs, depth + np.int64(1), s)
            candidates_mask ^= lsb
        return total

    @_njit(cache=True)
    def _nb_count_ks_cliques(adj, n, s):
        s64 = np.int64(s)
        if s64 <= np.int64(0):
            return np.int64(1)
        if s64 == np.int64(1):
            return np.int64(n)
        if s64 == np.int64(2):
            total = np.int64(0)
            for i in range(n):
                total += _nb_popcount(adj[i])
            return total >> np.int64(1)
        all_mask = (np.int64(1) << np.int64(n)) - np.int64(1)
        return _nb_clique_rec(adj, all_mask, np.int64(0), s64)

    @_njit(cache=True)
    def _nb_count_cliques_through_edge(adj, i, j, s):
        """Count K_s cliques containing edge (i,j). adj must already contain that edge."""
        if s <= np.int64(1):
            return np.int64(0)
        if s == np.int64(2):
            return np.int64(1)
        ij_mask = (np.int64(1) << i) | (np.int64(1) << j)
        common = adj[i] & adj[j] & ~ij_mask
        if s == np.int64(3):
            return _nb_popcount(common)
        return _nb_clique_rec(adj, common, np.int64(0), s - np.int64(2))

    @_njit(cache=True)
    def _nb_score_delta(adj, cadj, i, j, s, t):
        """Exact score delta for flipping edge (i,j). Positive means improvement."""
        bit_i = np.int64(1) << i
        bit_j = np.int64(1) << j
        if (adj[i] >> j) & np.int64(1):
            # Edge exists → removing it loses ks cliques, gains kt cliques in complement
            ks = _nb_count_cliques_through_edge(adj, i, j, s)
            cadj[i] |= bit_j
            cadj[j] |= bit_i
            kt = _nb_count_cliques_through_edge(cadj, i, j, t)
            cadj[i] ^= bit_j
            cadj[j] ^= bit_i
            return ks - kt
        else:
            # No edge → adding it gains ks cliques, loses kt cliques in complement
            adj[i] |= bit_j
            adj[j] |= bit_i
            ks = _nb_count_cliques_through_edge(adj, i, j, s)
            adj[i] ^= bit_j
            adj[j] ^= bit_i
            kt = _nb_count_cliques_through_edge(cadj, i, j, t)
            return kt - ks


def _popcount(x):
    return bin(x).count('1')


def count_ks_cliques_bitmask(adj, n, s):
    if s <= 0:
        return 1
    if s == 1:
        return n
    if s == 2:
        return sum(_popcount(adj[i]) for i in range(n)) // 2

    def rec(candidates_mask, depth):
        if depth == s:
            return 1
        total = 0
        m = candidates_mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            higher_nbrs = adj[v] & candidates_mask & ~((1 << (v + 1)) - 1)
            if _popcount(higher_nbrs) >= s - depth - 1:
                total += rec(higher_nbrs, depth + 1)
            candidates_mask ^= lsb
        return total

    return rec((1 << n) - 1, 0)


def count_cliques_through_edge(adj, n, i, j, size):
    if size < 2:
        return 0
    if size == 2:
        return 1
    common = adj[i] & adj[j] & ~(1 << i) & ~(1 << j)
    if size == 3:
        return _popcount(common)
    common_verts = []
    m = common
    while m:
        lsb = m & (-m)
        common_verts.append(lsb.bit_length() - 1)
        m ^= lsb
    nc = len(common_verts)
    if nc < size - 2:
        return 0
    v_idx = {v: k for k, v in enumerate(common_verts)}
    restricted = [0] * nc
    for ki, v in enumerate(common_verts):
        mask = adj[v] & common
        rm = mask
        while rm:
            lsb = rm & (-rm)
            u = lsb.bit_length() - 1
            ku = v_idx[u]
            restricted[ki] |= 1 << ku
            rm ^= lsb
    return count_ks_cliques_bitmask(restricted, nc, size - 2)


class RamseyDataPoint(DataPoint):
    """
    Graph on N vertices. Score = C(N,s)+C(N,t) - (K_s cliques + K_t cliques in complement).
    Max score is achieved iff the graph is a valid (s,t,N)-Ramsey counterexample.
    """

    MAKE_OBJECT_CANONICAL = False
    S = 5
    T = 5
    GEN_LOCAL_SEARCH = True

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((self.N, self.N), dtype=np.uint8)
        self.adj = [0] * self.N
        self.cadj = [0] * self.N

        if init:
            triu = np.random.randint(0, 2, (self.N, self.N), dtype=np.uint8)
            triu = np.triu(triu, 1)
            self.data = triu + triu.T
            self._sync_from_data()
            if self.__class__.GEN_LOCAL_SEARCH:
                self.local_search(improve_with_local_search=True)
            self.calc_features()
            self.calc_score()

    def _sync_from_data(self):
        n = self.N
        full = np.int64((1 << n) - 1)
        powers = np.int64(1) << np.arange(n, dtype=np.int64)
        adj_np = (self.data.astype(np.int64) @ powers)  # row dot powers = bitmask per row
        no_self = ~powers & full
        cadj_np = (~adj_np) & full & no_self
        self.adj = adj_np.tolist()
        self.cadj = cadj_np.tolist()

    def _flip_edge(self, i, j):
        if self.data[i, j]:
            self.data[i, j] = 0
            self.data[j, i] = 0
            self.adj[i] &= ~(1 << j)
            self.adj[j] &= ~(1 << i)
            self.cadj[i] |= (1 << j)
            self.cadj[j] |= (1 << i)
        else:
            self.data[i, j] = 1
            self.data[j, i] = 1
            self.adj[i] |= (1 << j)
            self.adj[j] |= (1 << i)
            self.cadj[i] &= ~(1 << j)
            self.cadj[j] &= ~(1 << i)

    def _score_delta_for_flip(self, i, j):
        s = self.__class__.S
        t = self.__class__.T
        if self.data[i, j]:
            ks_delta = -count_cliques_through_edge(self.adj, self.N, i, j, s)
            self.cadj[i] |= (1 << j)
            self.cadj[j] |= (1 << i)
            kt_delta = +count_cliques_through_edge(self.cadj, self.N, i, j, t)
            self.cadj[i] &= ~(1 << j)
            self.cadj[j] &= ~(1 << i)
        else:
            self.adj[i] |= (1 << j)
            self.adj[j] |= (1 << i)
            ks_delta = +count_cliques_through_edge(self.adj, self.N, i, j, s)
            self.adj[i] &= ~(1 << j)
            self.adj[j] &= ~(1 << i)
            kt_delta = -count_cliques_through_edge(self.cadj, self.N, i, j, t)
        return -(ks_delta + kt_delta)

    def calc_score(self):
        self._sync_from_data()
        s = self.__class__.S
        t = self.__class__.T
        max_score = math.comb(self.N, s) + math.comb(self.N, t)
        if _NUMBA:
            adj_arr = np.array(self.adj, dtype=np.int64)
            cadj_arr = np.array(self.cadj, dtype=np.int64)
            ks = int(_nb_count_ks_cliques(adj_arr, self.N, s))
            kt = int(_nb_count_ks_cliques(cadj_arr, self.N, t))
        else:
            ks = count_ks_cliques_bitmask(self.adj, self.N, s)
            kt = count_ks_cliques_bitmask(self.cadj, self.N, t)
        self.score = max_score - (ks + kt)

    def calc_features(self):
        idx = np.triu_indices(self.N, k=1)
        self.features = ",".join(map(str, self.data[idx].tolist()))

    def local_search(self, improve_with_local_search):
        n = self.N
        s = self.__class__.S
        t = self.__class__.T
        max_score = math.comb(n, s) + math.comb(n, t)
        self._sync_from_data()

        if _NUMBA:
            adj_arr = np.array(self.adj, dtype=np.int64)
            cadj_arr = np.array(self.cadj, dtype=np.int64)
            ks = int(_nb_count_ks_cliques(adj_arr, n, s))
            kt = int(_nb_count_ks_cliques(cadj_arr, n, t))
        else:
            ks = count_ks_cliques_bitmask(self.adj, n, s)
            kt = count_ks_cliques_bitmask(self.cadj, n, t)
        current_score = max_score - (ks + kt)
        self.score = current_score

        if current_score == max_score:
            return

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng = np.random.default_rng()

        # Phase 1: greedy hill climbing
        improved = True
        while improved and self.score < max_score:
            improved = False
            for (i, j) in all_pairs:
                delta = self._score_delta_for_flip(i, j)
                if delta > 0:
                    self._flip_edge(i, j)
                    self.score += delta
                    improved = True

        if self.score == max_score or not improve_with_local_search:
            return

        # Phase 2: simulated annealing
        violations = max_score - self.score
        T = max(0.5, violations * 0.1)
        T_min = 0.01
        max_sa_steps = n * n * 10
        cooling = (T_min / T) ** (1.0 / max(1, max_sa_steps))
        n_pairs = len(all_pairs)

        for _ in range(max_sa_steps):
            if self.score == max_score:
                break
            T *= cooling
            idx = rng.integers(n_pairs)
            i, j = all_pairs[idx]
            delta = self._score_delta_for_flip(i, j)
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                self._flip_edge(i, j)
                self.score += delta

    def local_search_fast(self, sa_steps=None):
        """
        Local search using numba full-score recompute instead of pure Python delta.
        ~50x faster than local_search() for large N.
        """
        if not _NUMBA:
            self.local_search(improve_with_local_search=True)
            return

        n = self.N
        s = self.__class__.S
        t = self.__class__.T
        max_score = math.comb(n, s) + math.comb(n, t)
        self._sync_from_data()

        adj_arr = np.array(self.adj, dtype=np.int64)
        cadj_arr = np.array(self.cadj, dtype=np.int64)

        def _recompute():
            return max_score - int(_nb_count_ks_cliques(adj_arr, n, s)) - int(_nb_count_ks_cliques(cadj_arr, n, t))

        def _flip(i, j):
            self._flip_edge(i, j)
            adj_arr[i] = self.adj[i];  adj_arr[j] = self.adj[j]
            cadj_arr[i] = self.cadj[i]; cadj_arr[j] = self.cadj[j]

        self.score = _recompute()
        if self.score == max_score:
            return

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        # Phase 1: greedy hill climbing
        improved = True
        while improved and self.score < max_score:
            improved = False
            for (i, j) in all_pairs:
                _flip(i, j)
                new_score = _recompute()
                if new_score > self.score:
                    self.score = new_score
                    improved = True
                else:
                    _flip(i, j)  # unflip

        if self.score == max_score:
            self.calc_features()
            return

        # Phase 2: simulated annealing
        if sa_steps is None:
            sa_steps = n * n * 10
        violations = max_score - self.score
        T = max(0.5, violations * 0.1)
        T_min = 0.01
        cooling = (T_min / T) ** (1.0 / max(1, sa_steps))
        rng = np.random.default_rng()
        n_pairs = len(all_pairs)

        for _ in range(sa_steps):
            if self.score == max_score:
                break
            T *= cooling
            i, j = all_pairs[int(rng.integers(n_pairs))]
            _flip(i, j)
            new_score = _recompute()
            delta = new_score - self.score
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                self.score = new_score
            else:
                _flip(i, j)  # unflip

        self.calc_features()

    def local_search_fast_v2(self, sa_steps=None, time_limit=None):
        """
        Delta-based local search: uses _nb_score_delta instead of full recompute per step.
        Expected ~20-50x faster than local_search_fast for large N.
        """
        if not _NUMBA:
            self.local_search(improve_with_local_search=True)
            return

        n = self.N
        s = self.__class__.S
        t = self.__class__.T
        s64 = np.int64(s)
        t64 = np.int64(t)
        max_score = math.comb(n, s) + math.comb(n, t)
        self._sync_from_data()

        adj_arr = np.array(self.adj, dtype=np.int64)
        cadj_arr = np.array(self.cadj, dtype=np.int64)

        self.score = max_score - int(_nb_count_ks_cliques(adj_arr, n, s)) - int(_nb_count_ks_cliques(cadj_arr, n, t))
        if self.score == max_score:
            return

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def _flip(i, j):
            self._flip_edge(i, j)
            adj_arr[i] = self.adj[i]; adj_arr[j] = self.adj[j]
            cadj_arr[i] = self.cadj[i]; cadj_arr[j] = self.cadj[j]

        # Phase 1: greedy hill climbing (no full recompute needed)
        improved = True
        while improved and self.score < max_score:
            improved = False
            for (i, j) in all_pairs:
                delta = int(_nb_score_delta(adj_arr, cadj_arr, np.int64(i), np.int64(j), s64, t64))
                if delta > 0:
                    _flip(i, j)
                    self.score += delta
                    improved = True
                    if self.score == max_score:
                        break

        if self.score == max_score:
            self.calc_features()
            return

        # Phase 2: simulated annealing
        if sa_steps is None:
            sa_steps = n * n * 10
        violations = max_score - self.score
        T = max(0.5, violations * 0.1)
        T_min = 0.01
        cooling = (T_min / T) ** (1.0 / max(1, sa_steps))
        rng = np.random.default_rng()
        n_pairs = len(all_pairs)
        deadline = (time.monotonic() + time_limit) if time_limit is not None else None

        for step in range(sa_steps):
            if self.score == max_score:
                break
            if deadline is not None and (step & 0xFFF) == 0 and time.monotonic() >= deadline:
                break
            T *= cooling
            i, j = all_pairs[int(rng.integers(n_pairs))]
            delta = int(_nb_score_delta(adj_arr, cadj_arr, np.int64(i), np.int64(j), s64, t64))
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                _flip(i, j)
                self.score += delta

        self.calc_features()

    @classmethod
    def max_possible_score(cls, N):
        return math.comb(N, cls.S) + math.comb(N, cls.T)

    @classmethod
    def _nb_warmup(cls):
        """Trigger numba JIT compilation in the main process so forked workers inherit it."""
        if not _NUMBA:
            return
        dummy = np.zeros(6, dtype=np.int64)
        dummy[0] = 0b110010
        dummy[1] = 0b100001
        _nb_count_ks_cliques(dummy, 6, cls.S)
        _nb_count_ks_cliques(dummy, 6, cls.T)
        cdummy = np.zeros(6, dtype=np.int64)
        cdummy[0] = ~dummy[0] & np.int64((1 << 6) - 1)
        cdummy[1] = ~dummy[1] & np.int64((1 << 6) - 1)
        _nb_score_delta(dummy, cdummy, np.int64(0), np.int64(1), np.int64(cls.S), np.int64(cls.T))

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL, cls.S, cls.T, cls.GEN_LOCAL_SEARCH = pars

    @classmethod
    def _save_class_params(cls):
        return cls.MAKE_OBJECT_CANONICAL, cls.S, cls.T, cls.GEN_LOCAL_SEARCH


class RamseyEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = RamseyDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.S = params.ramsey_s
        self.data_class.T = params.ramsey_t
        self.data_class.GEN_LOCAL_SEARCH = params.gen_local_search
        encoding_augmentation = random_symmetry_adj_matrix if params.augment_data_representation else None
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS, encoding_augmentation=encoding_augmentation,
            )
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(
                self.data_class, params.N, self.k, self.are_coordinates_symmetric,
                self.SPECIAL_SYMBOLS, pow2base=params.pow2base,
                encoding_augmentation=encoding_augmentation,
            )
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")

    @staticmethod
    def register_args(parser):
        parser.add_argument("--N", type=int, default=43, help="Number of vertices")
        parser.add_argument("--ramsey_s", type=int, default=5, help="Clique size to avoid")
        parser.add_argument("--ramsey_t", type=int, default=5, help="Independent set size to avoid")
        parser.add_argument("--encoding_tokens", type=str, default="single_integer",
                            help="single_integer / sequence_k_tokens / adjacency")
        parser.add_argument("--make_object_canonical", type=bool_flag, default="false",
                            help="Sort graph nodes by degree")
        parser.add_argument("--augment_data_representation", type=bool_flag, default="false",
                            help="Random vertex relabelling augmentation")
        parser.add_argument("--pow2base", type=int, default=1,
                            help="Adjacency entries packed together (adjacency encoding only)")
        parser.add_argument("--gen_local_search", type=bool_flag, default="true",
                            help="Run local search on each generated example during initial data generation")
