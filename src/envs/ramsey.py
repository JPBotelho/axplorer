"""
Ramsey graph environment for Axplorer.

score(G) = C(N,s) + C(N,t) - (count_Ks_in_G + count_Kt_in_complement_G)

Score is always >= 0. Max score = C(N,s) + C(N,t) iff G is a valid (s,t,N)-Ramsey graph.
Local search: greedy hill climbing + simulated annealing.
"""

import math
import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix
from src.utils import bool_flag


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
            self.calc_features()
            self.calc_score()

    def _sync_from_data(self):
        full = (1 << self.N) - 1
        self.adj = [0] * self.N
        for i in range(self.N):
            mask = 0
            for j in range(self.N):
                if self.data[i, j]:
                    mask |= 1 << j
            self.adj[i] = mask
        self.cadj = [(~self.adj[i]) & full & ~(1 << i) for i in range(self.N)]

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
        ks = count_ks_cliques_bitmask(self.adj, self.N, s)
        kt = count_ks_cliques_bitmask(self.cadj, self.N, t)
        max_score = math.comb(self.N, s) + math.comb(self.N, t)
        self.score = max_score - (ks + kt)

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    def local_search(self, improve_with_local_search):
        n = self.N
        s = self.__class__.S
        t = self.__class__.T
        max_score = math.comb(n, s) + math.comb(n, t)
        self._sync_from_data()

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

    @classmethod
    def _update_class_params(cls, pars):
        cls.MAKE_OBJECT_CANONICAL, cls.S, cls.T = pars

    @classmethod
    def _save_class_params(cls):
        return cls.MAKE_OBJECT_CANONICAL, cls.S, cls.T


class RamseyEnvironment(BaseEnvironment):
    k = 2
    are_coordinates_symmetric = True
    data_class = RamseyDataPoint

    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        self.data_class.S = params.ramsey_s
        self.data_class.T = params.ramsey_t
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
