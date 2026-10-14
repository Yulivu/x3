"""Nuisance-subspace projection (NEW_TASKS 02, U9 / N6).

Speaker and platform identity are metadata on *every* clip - they are known
whether the clip is genuine or fake - so the directions that predict them can be
removed from the feature block before a probe is fitted.  The directions come from
an LDA whose classes are the speaker ids (or the platforms); the top ``m`` are
orthonormalised and projected out:

    z_proj = z - (z W^T) W

Two fitting scopes exist and are always labelled:

* **inductive** - fit on the labelled speakers only, inside ``L`` (so the test
  speaker's identity never influences the projection);
* **transductive** - fit on every dev clip's metadata (audio-only, no labels).

The speaker and platform directions are not mutually orthogonal, so the platform
basis is Gram-Schmidt'ed against the speaker basis first: removing "the first
``m_s`` speaker directions and the first ``m_p`` platform directions" is then a
single, well-defined subspace.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "lda_directions", "orthonormalise", "orthogonalise_against", "residualise",
    "nuisance_basis",
]


def lda_directions(X: np.ndarray, groups: Sequence[object], n_components: int
                   ) -> np.ndarray:
    """Top ``n_components`` LDA directions of ``groups`` in ``X``, orthonormal rows."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    X = np.asarray(X, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k = int(min(int(n_components), len(uniq) - 1, X.shape[1]))
    if k <= 0:
        return np.zeros((0, X.shape[1]))
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=k).fit(X, groups)
    # sklearn's svd solver keeps more scaling columns than requested on some
    # versions, so the truncation is done here; the columns are already ordered by
    # decreasing discriminative importance.
    scalings = np.asarray(lda.scalings_, float)
    W = scalings[:, :k].T                           # (k, d)
    return orthonormalise(W)


def orthonormalise(W: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the row space of ``W``, **in the input order**.

    Ordered Gram-Schmidt, not SVD: the caller slices a prefix (``first m_s speaker
    directions``), so the row order has to be the LDA importance order.  An SVD
    would rotate within the row space and make "the first m" meaningless.
    """
    W = np.atleast_2d(np.asarray(W, float))
    if W.size == 0:
        return W.reshape(0, W.shape[1] if W.ndim == 2 else 0)
    out: list = []
    for row in W:
        v = row.copy()
        for u in out:
            v -= float(v @ u) * u
        n = float(np.linalg.norm(v))
        if n > 1e-10 * max(1.0, float(np.linalg.norm(row))):
            out.append(v / n)
    if not out:                                         # pragma: no cover
        return np.zeros((0, W.shape[1]))
    return np.ascontiguousarray(np.asarray(out))


def orthogonalise_against(W: np.ndarray, basis: Optional[np.ndarray]) -> np.ndarray:
    """``W`` with every component of ``basis`` removed, then orthonormalised."""
    W = np.atleast_2d(np.asarray(W, float))
    if W.size == 0:
        return W
    if basis is not None and np.asarray(basis).size:
        B = np.asarray(basis, float)
        W = W - (W @ B.T) @ B
    return orthonormalise(W)


def residualise(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project ``X`` onto the orthogonal complement of the rows of ``W``."""
    X = np.asarray(X, float)
    W = np.asarray(W, float)
    if W.size == 0:
        return X.copy()
    return np.asarray(X - (X @ W.T) @ W, dtype=np.float32)


def nuisance_basis(X: np.ndarray, speaker: Sequence[object], platform: Optional[Sequence[object]],
                   m_s: int, m_p: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The joint ``(m_s speaker + m_p platform)`` subspace, platform made orthogonal."""
    X = np.asarray(X, float)
    speaker = np.asarray(speaker)
    W_s = lda_directions(X, speaker, m_s)
    W_p = np.zeros((0, X.shape[1]))
    if platform is not None and int(m_p) > 0:
        plat = np.asarray(platform)
        if len(np.unique(plat)) > 1:
            W_p = orthogonalise_against(lda_directions(X, plat, m_p), W_s)
    W = np.vstack([W_s, W_p]) if W_p.size else W_s
    return W, {"m_s_available": int(W_s.shape[0]), "m_p_available": int(W_p.shape[0]),
               "dim_removed": int(W.shape[0]), "n_features": int(X.shape[1])}
