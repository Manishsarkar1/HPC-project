#!/usr/bin/env python3
"""
Distributed Graph Processing with MPI4Py
BFS, PageRank and Connected Components on a partitioned graph.

Run:  mpiexec -n 4 python dgp.py --scale 18 --partition hash --verify
"""
import argparse
import time

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank, P = comm.Get_rank(), comm.Get_size()
MPI_TYPES = {np.dtype("int64"): MPI.INT64_T, np.dtype("float64"): MPI.DOUBLE}


# ------------------------------------------------------------------ stats
class Stats:
    """Accumulates time spent inside MPI calls and bytes sent to other ranks."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.comm_t, self.nbytes = 0.0, 0


STATS = Stats()


def allreduce(x, op=MPI.SUM):
    t = time.perf_counter()
    r = comm.allreduce(x, op=op)
    STATS.comm_t += time.perf_counter() - t
    return r


def exchange_counts(scounts):
    t = time.perf_counter()
    rcounts = np.array(comm.alltoall(scounts.tolist()), dtype=np.int64)
    STATS.comm_t += time.perf_counter() - t
    return rcounts


def exchange(arr, scounts, rcounts):
    """Alltoallv for one 1-D array that is already sorted by destination rank."""
    sdisp = np.concatenate(([0], np.cumsum(scounts)[:-1]))
    rdisp = np.concatenate(([0], np.cumsum(rcounts)[:-1]))
    out = np.empty(int(rcounts.sum()), dtype=arr.dtype)
    mt = MPI_TYPES[arr.dtype]
    t = time.perf_counter()
    comm.Alltoallv([arr, (scounts.tolist(), sdisp.tolist()), mt],
                   [out, (rcounts.tolist(), rdisp.tolist()), mt])
    STATS.comm_t += time.perf_counter() - t
    STATS.nbytes += int(arr.nbytes - scounts[rank] * arr.itemsize)  # off-rank bytes only
    return out


# ------------------------------------------------------------ partitioning
class Partition:
    """1-D vertex partition. 'range' = contiguous blocks, 'hash' = v mod P."""

    def __init__(self, n, kind):
        self.n, self.kind = n, kind
        self.block = -(-n // P)
        if kind == "range":
            lo = min(n, rank * self.block)
            hi = min(n, lo + self.block)
            self.gids = np.arange(lo, hi, dtype=np.int64)
        else:
            self.gids = np.arange(rank, n, P, dtype=np.int64)
        self.n_local = len(self.gids)

    def owner(self, v):
        return v // self.block if self.kind == "range" else v % P

    def local(self, v):
        return v - rank * self.block if self.kind == "range" else v // P


class LocalGraph:
    """CSR of the owned vertices + a reusable ghost-exchange schedule."""

    def __init__(self, part, src, dst):
        ls = part.local(src)
        order = np.argsort(ls, kind="stable")
        self.dst, self.src_l = dst[order], ls[order]
        self.deg = np.bincount(ls, minlength=part.n_local)
        self.indptr = np.zeros(part.n_local + 1, dtype=np.int64)
        np.cumsum(self.deg, out=self.indptr[1:])
        self.n_edges = len(dst)

        # ghost schedule: unique remote targets, grouped by owner rank.
        ud, inv = np.unique(self.dst, return_inverse=True)
        inv = inv.ravel()
        own = part.owner(ud)
        self.ud_order = np.argsort(own, kind="stable")
        self.scounts = np.bincount(own, minlength=P)
        self.rcounts = exchange_counts(self.scounts)
        recv_ids = exchange(ud[self.ud_order], self.scounts, self.rcounts)  # sent once
        # sender side: edges grouped by target -> segmented reduce
        self.edge_perm = np.argsort(inv, kind="stable")
        self.edge_starts = np.flatnonzero(np.diff(inv[self.edge_perm], prepend=-1))
        # receiver side: incoming ids grouped by local vertex
        recv_l = part.local(recv_ids)
        self.rperm = np.argsort(recv_l, kind="stable")
        rs = recv_l[self.rperm]
        self.rstarts = np.flatnonzero(np.diff(rs, prepend=-1))
        self.runique = rs[self.rstarts]

    def push(self, vertex_vals, ufunc):
        """Send each vertex's value along its edges; combine at the target with
        ufunc (np.add / np.minimum). Returns (local_ids, combined_values)."""
        ev = vertex_vals[self.src_l][self.edge_perm]
        agg = ufunc.reduceat(ev, self.edge_starts) if len(ev) else ev
        recv = exchange(agg[self.ud_order], self.scounts, self.rcounts)
        if len(recv) == 0:
            return self.runique, recv
        return self.runique, ufunc.reduceat(recv[self.rperm], self.rstarts)


def expand(indptr, indices, verts):
    """All neighbours of `verts` (vectorised CSR slice gather)."""
    if len(verts) == 0:
        return indices[:0]
    starts = indptr[verts]
    counts = indptr[verts + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return indices[:0]
    offs = np.cumsum(counts) - counts
    return indices[np.arange(total) - np.repeat(offs - starts, counts)]


# -------------------------------------------------------- distributed algos
def bfs(g, part, source):
    dist = np.full(part.n_local, -1, dtype=np.int64)
    frontier = np.empty(0, dtype=np.int64)
    if part.owner(source) == rank:
        l = int(part.local(source))
        dist[l], frontier = 0, np.array([l], dtype=np.int64)
    level = 0
    while allreduce(len(frontier)) > 0:                       # Allreduce: termination
        nbrs = np.unique(expand(g.indptr, g.dst, frontier))
        own = part.owner(nbrs)
        order = np.argsort(own, kind="stable")
        sc = np.bincount(own, minlength=P)
        rc = exchange_counts(sc)
        recv = exchange(nbrs[order], sc, rc)                  # Alltoallv: ghost vertices
        cand = np.unique(part.local(recv))
        frontier = cand[dist[cand] < 0]
        dist[frontier] = level + 1
        level += 1
    return dist, level


def pagerank(g, part, n, d=0.85, tol=1e-8, max_iter=100):
    pr = np.full(part.n_local, 1.0 / n)
    has_out = g.deg > 0
    inv_deg = np.zeros(part.n_local)
    inv_deg[has_out] = 1.0 / g.deg[has_out]
    for it in range(1, max_iter + 1):
        idx, red = g.push(pr * inv_deg, np.add)
        incoming = np.zeros(part.n_local)
        incoming[idx] = red
        dangling = allreduce(float(pr[~has_out].sum()))
        new = (1 - d) / n + d * (incoming + dangling / n)
        diff = allreduce(float(np.abs(new - pr).sum()))       # Allreduce: convergence
        pr = new
        if diff < tol:
            break
    return pr, it


def connected_components(g, part):
    label = part.gids.copy()
    it = 0
    while True:
        it += 1
        idx, red = g.push(label, np.minimum)
        new = label.copy()
        if len(idx):
            new[idx] = np.minimum(label[idx], red)
        changed = allreduce(int(np.count_nonzero(new != label)))
        label = new
        if changed == 0:
            return label, it


# ---------------------------------------------------- serial baselines (rank 0)
def build_csr(n, src, dst):
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order]


def serial_bfs(n, csr, source):
    indptr, indices = csr
    dist = np.full(n, -1, dtype=np.int64)
    dist[source] = 0
    frontier, level = np.array([source], dtype=np.int64), 0
    while len(frontier):
        nb = np.unique(expand(indptr, indices, frontier))
        nb = nb[dist[nb] < 0]
        level += 1
        dist[nb], frontier = level, nb
    return dist, level


def serial_pagerank(n, src, dst, d=0.85, tol=1e-8, max_iter=100):
    deg = np.bincount(src, minlength=n)
    inv = np.zeros(n)
    inv[deg > 0] = 1.0 / deg[deg > 0]
    pr = np.full(n, 1.0 / n)
    for it in range(1, max_iter + 1):
        incoming = np.bincount(dst, weights=(pr * inv)[src], minlength=n)
        new = (1 - d) / n + d * (incoming + pr[deg == 0].sum() / n)
        diff = np.abs(new - pr).sum()
        pr = new
        if diff < tol:
            break
    return pr, it


def cc_prep(src, dst):
    """Sort edges by target once (excluded from timing, like the parallel schedule)."""
    perm = np.argsort(dst, kind="stable")
    s, d = src[perm], dst[perm]
    starts = np.flatnonzero(np.diff(d, prepend=-1))
    return s, starts, d[starts]


def serial_cc(n, prep):
    """Same min-label propagation as the distributed version, on one process."""
    s, starts, targets = prep
    label = np.arange(n, dtype=np.int64)
    it = 0
    while True:
        it += 1
        new = label.copy()
        new[targets] = np.minimum(label[targets], np.minimum.reduceat(label[s], starts))
        if np.array_equal(new, label):
            return label, it
        label = new


def scipy_component_count(n, src, dst):
    """Independent check (optional): number of components according to SciPy."""
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ImportError:
        return None
    m = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    return connected_components(m, directed=False)[0]


# ------------------------------------------------------------ graph input
def make_graph(args):
    """Rank 0 only. Returns n and a symmetrised edge list (src, dst)."""
    rng = np.random.default_rng(args.seed)
    if args.input:
        e = np.loadtxt(args.input, dtype=np.int64).reshape(-1, 2)
        u, v, n = e[:, 0], e[:, 1], int(e.max()) + 1
    else:  # R-MAT: skewed degrees, like real social/web graphs
        n = 1 << args.scale
        m = n * args.edge_factor
        a, b, c = 0.57, 0.19, 0.19
        u = np.zeros(m, dtype=np.int64)
        v = np.zeros(m, dtype=np.int64)
        for _ in range(args.scale):
            r = rng.random(m)
            u = (u << 1) | (r >= a + b)
            v = (v << 1) | (((r >= a) & (r < a + b)) | (r >= a + b + c))
    if args.shuffle:                       # random relabelling spreads the hubs
        perm = rng.permutation(n).astype(np.int64)
        u, v = perm[u], perm[v]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    keep = lo != hi                                           # drop self loops
    keys = np.unique(lo[keep] * n + hi[keep])                 # drop duplicates
    lo, hi = keys // n, keys % n
    return n, np.concatenate([lo, hi]), np.concatenate([hi, lo])


def scatter_graph(n, src, dst, part_kind):
    """Rank 0 splits the edge list by owner(src) and Scatter()s it."""
    part = Partition(n, part_kind)
    chunks = None
    if rank == 0:
        own = part.owner(src)
        order = np.argsort(own, kind="stable")
        cuts = np.cumsum(np.bincount(own, minlength=P))[:-1]
        chunks = list(zip(np.split(src[order], cuts), np.split(dst[order], cuts)))
    ls, ld = comm.scatter(chunks, root=0)
    return part, LocalGraph(part, ls, ld)


# ------------------------------------------------------------------- main
def run_timed(fn):
    comm.Barrier()
    STATS.reset()
    t0 = time.perf_counter()
    out = fn()
    comm.Barrier()
    wall = time.perf_counter() - t0
    compute = wall - STATS.comm_t
    stats = comm.gather((compute, STATS.comm_t, STATS.nbytes), root=0)
    return out, wall, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=16, help="2^scale vertices (R-MAT)")
    ap.add_argument("--edge-factor", type=int, default=16)
    ap.add_argument("--input", help="edge-list text file: 'u v' per line")
    ap.add_argument("--shuffle", action="store_true", help="randomly relabel vertices")
    ap.add_argument("--partition", choices=["range", "hash"], default="hash")
    ap.add_argument("--algos", default="bfs,pagerank,cc")
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify", action="store_true", help="check vs serial results")
    ap.add_argument("--no-serial", action="store_true", help="skip serial baseline")
    args = ap.parse_args()
    algos = args.algos.split(",")

    t0 = time.perf_counter()
    n, src, dst = make_graph(args) if rank == 0 else (None, None, None)
    t_gen = time.perf_counter() - t0
    n = comm.bcast(n, root=0)
    comm.Barrier()
    t0 = time.perf_counter()
    part, g = scatter_graph(n, src, dst, args.partition)
    comm.Barrier()
    t_scatter = time.perf_counter() - t0

    edges = comm.gather(g.n_edges, root=0)
    if rank == 0:
        e = np.array(edges)
        print(f"graph: n={n:,} undirected_edges={len(src)//2:,} | P={P} partition={args.partition}")
        print(f"setup: generate={t_gen:.2f}s scatter+schedule={t_scatter:.2f}s | "
              f"edges/rank min={e.min():,} max={e.max():,} imbalance={e.max()/e.mean():.2f}")
        need = not args.no_serial or args.verify
        csr = build_csr(n, src, dst) if need else None
        prep = cc_prep(src, dst) if need else None

    jobs = {
        "bfs": lambda: bfs(g, part, args.source),
        "pagerank": lambda: pagerank(g, part, n),
        "cc": lambda: connected_components(g, part),
    }
    serial = {}
    if rank == 0:
        serial = {
            "bfs": lambda: serial_bfs(n, csr, args.source),
            "pagerank": lambda: serial_pagerank(n, src, dst),
            "cc": lambda: serial_cc(n, prep),
        }

    for name in algos:
        (res, iters), wall, stats = run_timed(jobs[name])

        if name == "bfs":
            info = f"levels={iters} reached={comm.allreduce(int((res >= 0).sum())):,}"
        elif name == "cc":
            info = f"components={comm.allreduce(int((res == part.gids).sum())):,} rounds={iters}"
        else:
            info = f"iterations={iters}"

        full = None
        if args.verify:
            parts = comm.gather((part.gids, res), root=0)
            if rank == 0:
                full = np.empty(n, dtype=res.dtype)
                for gids, vals in parts:
                    full[gids] = vals

        t_serial, verdict = float("nan"), ""
        comm.Barrier()
        if rank == 0 and (not args.no_serial or args.verify):
            t0 = time.perf_counter()
            ref = serial[name]()
            t_serial = time.perf_counter() - t0
            if args.verify:
                if name == "bfs":
                    ok = np.array_equal(full, ref[0])
                elif name == "pagerank":
                    ok = np.allclose(full, ref[0], rtol=1e-5, atol=1e-12)
                else:
                    ok = np.array_equal(full, ref[0])
                    sc = scipy_component_count(n, src, dst)
                    if sc is not None:
                        ok = ok and sc == len(np.unique(full))
                verdict = " | verify " + ("PASS" if ok else "FAIL")
        comm.Barrier()

        if rank == 0:
            comp = np.array([s[0] for s in stats])
            cm = np.array([s[1] for s in stats])
            mb = sum(s[2] for s in stats) / 1e6
            imb = comp.max() / max(comp.mean(), 1e-12)
            pct = 100 * cm.mean() / wall
            speed = t_serial / wall
            ser = "serial n/a | speedup n/a" if t_serial != t_serial else \
                  f"serial {t_serial:.3f}s | speedup {speed:.2f}x"
            print(f"[{name}] {info} | parallel {wall:.3f}s (comm {pct:.0f}%, {mb:.1f} MB sent) | "
                  f"{ser} | "
                  f"compute imbalance {imb:.2f}{verdict}")
            print(f"RESULT,{P},{args.partition},{name},{wall:.4f},{cm.mean():.4f},"
                  f"{t_serial:.4f},{speed:.3f},{imb:.3f},{mb:.2f}")


if __name__ == "__main__":
    main()