"""Parallel-axis mass + inertia combination for weld collapse (Stage Z).

Pure numpy — no OpenSim dependency, fully unit-testable against analytic cases.

When an intermediate ``WeldJoint`` is collapsed, the placeholder sub-body B is
absorbed into the main body A. Because Stage Z only handles translation-only
welds, A and B share frame axes (no relative rotation), so inertia tensors need
no rotation — only a parallel-axis shift to the combined mass center.
"""

import numpy as np

__all__ = ["inertia_6vec_to_matrix", "inertia_matrix_to_6vec", "combine_inertia"]


def inertia_6vec_to_matrix(vec: np.ndarray) -> np.ndarray:
    """Expand an OpenSim inertia 6-vector to a symmetric 3x3 tensor.

    OpenSim stores ``<inertia>`` as ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``.
    """
    vec = np.asarray(vec, dtype=float)
    if vec.shape != (6,):
        raise ValueError(f"inertia 6-vector must have shape (6,), got {vec.shape}")
    ixx, iyy, izz, ixy, ixz, iyz = vec
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ],
        dtype=float,
    )


def inertia_matrix_to_6vec(mat: np.ndarray) -> np.ndarray:
    """Flatten a symmetric 3x3 inertia tensor to OpenSim's 6-vector ordering.

    Returns ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (3, 3):
        raise ValueError(f"inertia matrix must have shape (3, 3), got {mat.shape}")
    return np.array(
        [mat[0, 0], mat[1, 1], mat[2, 2], mat[0, 1], mat[0, 2], mat[1, 2]],
        dtype=float,
    )


def _parallel_axis_shift(inertia_about_com: np.ndarray, mass: float, r: np.ndarray) -> np.ndarray:
    """Shift a 3x3 inertia tensor from a body's own COM to a point displaced by ``r``.

    ``r`` is the displacement vector from the body's COM to the target point.
    I_P = I_COM + m * (|r|^2 * E - r (x) r).
    """
    r = np.asarray(r, dtype=float)
    return inertia_about_com + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def combine_inertia(
    m_a: float,
    com_a: np.ndarray,
    inertia_a: np.ndarray,
    m_b: float,
    com_b: np.ndarray,
    inertia_b: np.ndarray,
    d: np.ndarray,
):
    """Combine placeholder body B into main body A (no relative rotation).

    Parameters
    ----------
    m_a, m_b
        Body masses (kg).
    com_a, com_b
        Body mass centers (3-vectors), each expressed in its own body frame.
    inertia_a, inertia_b
        Inertia 6-vectors ``(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)``, each taken about
        that body's own mass center, expressed in its own body frame.
    d
        Translation from the B-body frame to the A-body frame (3-vector). For a
        translation-only weld this is the full rigid transform.

    Returns
    -------
    (combined_mass, combined_com, combined_inertia_6vec)
        ``combined_mass`` is ``m_a + m_b``; ``combined_com`` is the combined
        mass center expressed in the A-body frame; ``combined_inertia_6vec`` is
        the inertia 6-vector about that combined mass center, in the A frame.
    """
    com_a = np.asarray(com_a, dtype=float)
    com_b = np.asarray(com_b, dtype=float)
    d = np.asarray(d, dtype=float)

    total_mass = m_a + m_b
    if total_mass <= 0.0:
        raise ValueError(f"combined mass must be positive, got {total_mass}")

    # B's mass center re-expressed in the A frame (axes are parallel).
    com_b_in_a = com_b + d
    combined_com = (m_a * com_a + m_b * com_b_in_a) / total_mass

    i_a = inertia_6vec_to_matrix(inertia_a)
    i_b = inertia_6vec_to_matrix(inertia_b)

    # Shift each body's inertia to be about the combined mass center.
    i_a_at_c = _parallel_axis_shift(i_a, m_a, combined_com - com_a)
    i_b_at_c = _parallel_axis_shift(i_b, m_b, combined_com - com_b_in_a)
    i_total = i_a_at_c + i_b_at_c

    return total_mass, combined_com, inertia_matrix_to_6vec(i_total)
