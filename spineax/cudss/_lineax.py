# pyright: reportMissingImports=false
"""Lineax operator and solver adapters for the token API."""

import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.flatten_util as jfu
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array

from ._api import analyze, factorize, query, refactorize, solve
from ._transpose import _transpose_csr

# lineax front door — the default user-facing API ==============================


class CSROperator(lx.AbstractLinearOperator):
    """A square matrix in CSR form (full pattern; values as the one leaf).

    Structure is declared lineax-style via TAGS, not subclasses — exactly
    like ``lx.MatrixLinearOperator(matrix, lx.symmetric_tag)`` for dense:

        CSROperator(vals, offs, cols)                       # general
        CSROperator(vals, offs, cols, lx.symmetric_tag)     # symmetric
        CSROperator(vals, offs, cols,
                    (lx.symmetric_tag,
                     lx.positive_semidefinite_tag))         # SPD

    Column indices may be unsorted but must be unique within each row.

    Always the FULL sparsity pattern (both triangles; cuDSS ``mview_id=0``):
    ``mv`` is a plain BCSR matvec and a general matrix has no triangular
    shorthand anyway. The arrays are referenced zero-copy, same as
    everywhere else.
    """

    values: Array
    offsets: Array
    columns: Array
    tags: frozenset = eqx.field(static=True)

    def __init__(self, values, offsets, columns, tags=()):
        self.values = values
        self.offsets = offsets
        self.columns = columns
        if isinstance(tags, (tuple, list, set, frozenset)):
            self.tags = frozenset(tags)
        else:
            self.tags = frozenset({tags})

    def _bcsr(self):
        n = self.offsets.shape[0] - 1
        return jsparse.BCSR((self.values, self.columns, self.offsets), shape=(n, n))

    def mv(self, vector):
        return self._bcsr() @ vector

    def as_matrix(self):
        return self._bcsr().todense()

    def transpose(self):
        if lx.symmetric_tag in self.tags:
            return self
        t_vals, t_offs, t_cols = _transpose_csr(self.values, self.offsets, self.columns)
        return CSROperator(t_vals, t_offs, t_cols, lx.transpose_tags(self.tags))

    def in_structure(self):
        n = self.offsets.shape[0] - 1
        return jax.ShapeDtypeStruct((n,), self.values.dtype)

    def out_structure(self):
        return self.in_structure()


# lineax dispatches these predicates by operator class; each reads its tag,
# mirroring how lx.MatrixLinearOperator + tags behaves for dense matrices
for _predicate, _tag in (
    (lx.is_symmetric, lx.symmetric_tag),
    (lx.is_diagonal, lx.diagonal_tag),
    (lx.is_tridiagonal, lx.tridiagonal_tag),
    (lx.is_lower_triangular, lx.lower_triangular_tag),
    (lx.is_upper_triangular, lx.upper_triangular_tag),
    (lx.is_positive_semidefinite, lx.positive_semidefinite_tag),
    (lx.is_negative_semidefinite, lx.negative_semidefinite_tag),
    (lx.has_unit_diagonal, lx.unit_diagonal_tag),
):
    _predicate.register(CSROperator)(lambda operator, _tag=_tag: _tag in operator.tags)


@lx.linearise.register(CSROperator)
@lx.materialise.register(CSROperator)
def _(operator):
    return operator


@lx.conj.register(CSROperator)
def _(operator):
    return CSROperator(
        jnp.conj(operator.values), operator.offsets, operator.columns, operator.tags
    )


class CuDSS(lx.AbstractLinearSolver):
    """lineax front door for the token API.

    The cuDSS matrix type is resolved from the OPERATOR'S TAGS, the same
    way lineax's own solvers consult ``lx.is_symmetric`` etc.:

        symmetric + positive_semidefinite  ->  mtype 3 (Cholesky)
        symmetric                          ->  mtype 1 (LDL^T)
        untagged                           ->  mtype 0 (general LU)

    General operators are fully supported, gradients included — with the
    documented cost that anything needing ``A^T`` (lineax's backward pass,
    ``solver.transpose``) must factorize the transpose from scratch, since
    cuDSS has no transpose solve (design doc section 8).

    lineax's phase boundary is operator-dependent vs vector-dependent work
    (``lx.Cholesky.init`` runs ``cho_factor``; ``compute`` runs
    ``cho_solve``), so the protocol slots follow that convention:

        init    = analyze + factorize    (all operator-dependent work)
        compute = solve                  (per-vector work only)

    The state is the factorized token, so lineax's ``state=`` argument
    means the same thing it means for every built-in solver: one
    factorization, many right-hand sides.

    Every un-stated ``lx.linear_solve`` call re-inits, minting a registry
    entry whose cuDSS factors occupy device memory (the CSR arrays are
    zero-copy references in the token, never duplicated). Outside jit, free
    it eagerly with ``release(sol.state)``; under jit that is impossible
    and the LRU (``SPINEAX_FACTOR_CACHE``, default 8) bounds the leak by
    evicting oldest-used entries. Eviction is not fatal — a stated solve
    self-heals by re-factorizing from the token's own arrays (see
    ``rebuild_count``) — it just costs the repeated work.

    For true control the phases are explicit methods that thread tokens,
    mirroring the ``spineax.cudss`` free functions with operator sugar:

        solver = CuDSS()
        token  = solver.analyze(operator)                 # ANALYSIS
        token  = solver.factorize(token, operator)        # FACTORIZATION
        token  = solver.refactorize(token, new_operator)  # REFACTORIZATION
        x      = solver.solve(token, b)                   # SOLVE (repeatable)
        data   = solver.query(token)                      # every cuDSS data item
    """

    # cuDSS knobs (see ``analyze``): reordering algorithm and factor storage
    reordering: str = "default"
    memory: str = "device"

    # explicit phases ----------------------------------------------------------

    def analyze(self, operator):
        if lx.is_symmetric(operator):
            mtype_id = 3 if lx.is_positive_semidefinite(operator) else 1
        else:
            mtype_id = 0
        return analyze(
            operator.values,
            operator.offsets,
            operator.columns,
            mtype_id=mtype_id,
            mview_id=0,
            reordering=self.reordering,
            memory=self.memory,
        )

    def factorize(self, token, operator):
        return factorize(token, operator.values)

    def refactorize(self, token, operator):
        return refactorize(token, operator.values)

    def solve(self, token, vector, ir_nsteps=None):
        return solve(token, vector, ir_nsteps=ir_nsteps)

    def query(self, token):
        return query(token)

    # lineax protocol ----------------------------------------------------------

    def init(self, operator, options):
        del options
        return self.factorize(self.analyze(operator), operator)

    def compute(self, state, vector, options):
        # lineax's per-call channel: lx.linear_solve(..., options={"ir_nsteps": k})
        vector, unflatten = jfu.ravel_pytree(vector)
        solution = self.solve(state, vector, ir_nsteps=options.get("ir_nsteps"))
        return unflatten(solution), lx.RESULTS.successful, {}

    def transpose(self, state, options):
        if state.mtype_id in (1, 3):
            return state, options  # A^T = A: same factorization
        # general: cuDSS has no transpose solve, so lineax's backward pass
        # pays for a fresh factorization of A^T (one analyze+factorize+
        # registry entry), mirroring the raw autodiff path
        t_vals, t_offs, t_cols = _transpose_csr(
            state.values, state.offsets, state.columns
        )
        t_token = analyze(
            t_vals,
            t_offs,
            t_cols,
            mtype_id=0,
            mview_id=0,
            device_id=state.device_id,
            reordering=state.reordering_id,
            memory=state.memory_id,
        )
        return factorize(t_token, t_token.values), options

    def conj(self, state, options):
        if not jnp.issubdtype(state.dtype, jnp.complexfloating):
            return state, options
        values = jnp.conj(state.values)
        token = analyze(
            values,
            state.offsets,
            state.columns,
            mtype_id=state.mtype_id,
            mview_id=state.mview_id,
            device_id=state.device_id,
            reordering=state.reordering_id,
            memory=state.memory_id,
        )
        return factorize(token, values), options

    def assume_full_rank(self):
        return True
