import numpy as np

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import DenseTokenizer, SparseTokenizerSequenceKTokens, SparseTokenizerSingleInteger
from src.envs.utils import random_symmetry_adj_matrix, sort_graph_based_on_degree
from src.utils import bool_flag


def _popcount(x):
    return bin(x).count('1')


class RamseyDataPoint(DataPoint):
    """
    Graph on N vertices with no K_s clique and no independent set of size t.
    Searching for lower bounds on the Ramsey number R(s, t).
    Score = number of edges if valid, -1 otherwise.
    """

    MAKE_OBJECT_CANONICAL = False
    S = 5  # clique size to avoid
    T = 5  # independent set size to avoid

    def __init__(self, N, init=False):
        super().__init__()
        self.N = N
        self.data = np.zeros((self.N, self.N), dtype=np.uint8)

        if init:
            triu = np.random.randint(0, 2, (self.N, self.N), dtype=np.uint8)
            triu = np.triu(triu, 1)
            self.data = triu + triu.T
            self.local_search(improve_with_local_search=True)
            if self.MAKE_OBJECT_CANONICAL:
                self.data = sort_graph_based_on_degree(self.data)
            self.calc_features()
            self.calc_score()

    # ── Bitmask helpers ───────────────────────────────────────────────────────

    def _build_masks(self):
        masks = []
        for i in range(self.N):
            mask = 0
            for j in range(self.N):
                if self.data[i, j]:
                    mask |= 1 << j
            masks.append(mask)
        return masks

    def _build_comp_masks(self):
        masks = []
        for i in range(self.N):
            mask = 0
            for j in range(self.N):
                if i != j and self.data[i, j] == 0:
                    mask |= 1 << j
            masks.append(mask)
        return masks

    def _find_clique_k(self, masks, k):
        """Return first tuple of k mutually adjacent vertices, or None."""
        def recurse(chosen, cands, depth):
            if depth == k:
                return chosen
            tmp = cands
            while tmp:
                lsb = tmp & (-tmp)
                v = lsb.bit_length() - 1
                tmp ^= lsb
                new_cands = cands & masks[v] & ~((1 << (v + 1)) - 1)
                if _popcount(new_cands) >= k - depth - 1:
                    result = recurse(chosen + (v,), new_cands, depth + 1)
                    if result is not None:
                        return result
            return None

        for i in range(self.N):
            cands = masks[i] & ~((1 << (i + 1)) - 1)
            if _popcount(cands) >= k - 1:
                result = recurse((i,), cands, 1)
                if result is not None:
                    return result
        return None

    # ── Violation finders ─────────────────────────────────────────────────────

    def _find_ks(self):
        return self._find_clique_k(self._build_masks(), self.S)

    def _find_ist(self):
        return self._find_clique_k(self._build_comp_masks(), self.T)

    # ── Local search ──────────────────────────────────────────────────────────

    def local_search(self, improve_with_local_search):
        max_iter = 300
        for _ in range(max_iter):
            ks = self._find_ks()
            ist = self._find_ist()
            if ks is None and ist is None:
                break

            if ks is not None:
                # Remove the edge in the clique with the highest combined degree (greedy)
                verts = list(ks)
                best_edge, best_deg = None, -1
                for a in range(len(verts)):
                    for b in range(a + 1, len(verts)):
                        i, j = verts[a], verts[b]
                        deg = int(self.data[i].sum()) + int(self.data[j].sum())
                        if deg > best_deg:
                            best_deg = deg
                            best_edge = (i, j)
                if best_edge:
                    i, j = best_edge
                    self.data[i, j] = 0
                    self.data[j, i] = 0

            if ist is not None:
                # Add the edge in the IS with the lowest combined degree (greedy)
                verts = list(ist)
                best_edge, best_deg = None, float('inf')
                for a in range(len(verts)):
                    for b in range(a + 1, len(verts)):
                        i, j = verts[a], verts[b]
                        if self.data[i, j] == 0:
                            deg = int(self.data[i].sum()) + int(self.data[j].sum())
                            if deg < best_deg:
                                best_deg = deg
                                best_edge = (i, j)
                if best_edge:
                    i, j = best_edge
                    self.data[i, j] = 1
                    self.data[j, i] = 1

    # ── Score / features ──────────────────────────────────────────────────────

    def calc_score(self):
        if self._find_ks() is not None or self._find_ist() is not None:
            self.score = -1
        else:
            self.score = int(self.data.sum()) // 2

    def calc_features(self):
        w = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))

    # ── Multiprocessing param passing ─────────────────────────────────────────

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
