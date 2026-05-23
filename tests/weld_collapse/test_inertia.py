"""Unit tests for nsosim.weld_collapse.inertia — parallel-axis combination.

Pure numpy, no OpenSim. Checked against analytic point-mass / dumbbell cases.
"""

import numpy as np
import pytest

from nsosim.weld_collapse.inertia import (
    combine_inertia,
    inertia_6vec_to_matrix,
    inertia_matrix_to_6vec,
)

ZERO6 = np.zeros(6)


class TestInertia6vecConversion:
    def test_6vec_to_matrix_layout(self):
        mat = inertia_6vec_to_matrix(np.array([1, 2, 3, 4, 5, 6]))
        expected = np.array([[1, 4, 5], [4, 2, 6], [5, 6, 3]], dtype=float)
        np.testing.assert_array_equal(mat, expected)

    def test_roundtrip(self):
        vec = np.array([0.7, 1.1, 1.3, -0.2, 0.4, -0.05])
        np.testing.assert_allclose(
            inertia_matrix_to_6vec(inertia_6vec_to_matrix(vec)), vec, atol=1e-15
        )

    def test_matrix_is_symmetric(self):
        mat = inertia_6vec_to_matrix(np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(mat, mat.T)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            inertia_6vec_to_matrix(np.array([1, 2, 3]))
        with pytest.raises(ValueError):
            inertia_matrix_to_6vec(np.array([1, 2, 3]))


class TestCombineInertia:
    def test_dumbbell_two_point_masses(self):
        """Two equal point masses at +/- a along x (same frame, d=0).

        Classic dumbbell: I about the connecting axis (x) = 0; I about each
        perpendicular axis = 2 * m * a^2.
        """
        m, a = 2.0, 0.3
        mass, com, inertia = combine_inertia(
            m,
            np.array([-a, 0, 0]),
            ZERO6,
            m,
            np.array([a, 0, 0]),
            ZERO6,
            d=np.zeros(3),
        )
        assert mass == pytest.approx(2 * m)
        np.testing.assert_allclose(com, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(inertia, [0.0, 2 * m * a**2, 2 * m * a**2, 0, 0, 0], atol=1e-14)

    def test_translation_d_is_applied(self):
        """Same dumbbell, but B's frame is offset by d; result must be identical.

        com_b in B-frame is (-a,0,0); d=(2a,0,0) puts it at (a,0,0) in A-frame.
        """
        m, a = 2.0, 0.3
        mass, com, inertia = combine_inertia(
            m,
            np.array([-a, 0, 0]),
            ZERO6,
            m,
            np.array([-a, 0, 0]),
            ZERO6,
            d=np.array([2 * a, 0, 0]),
        )
        assert mass == pytest.approx(2 * m)
        np.testing.assert_allclose(com, np.zeros(3), atol=1e-15)
        np.testing.assert_allclose(inertia, [0.0, 2 * m * a**2, 2 * m * a**2, 0, 0, 0], atol=1e-14)

    def test_identity_when_b_is_massless(self):
        """m_b = 0 (with zero inertia) leaves body A exactly unchanged."""
        m_a = 8.275
        com_a = np.array([0.0, -0.150, 0.0])
        inertia_a = np.array([0.12, 0.03, 0.13, 0.001, -0.002, 0.004])
        mass, com, inertia = combine_inertia(
            m_a,
            com_a,
            inertia_a,
            0.0,
            np.array([1.0, 2.0, 3.0]),
            ZERO6,
            d=np.array([5.0, 6.0, 7.0]),
        )
        assert mass == pytest.approx(m_a)
        np.testing.assert_allclose(com, com_a, atol=1e-15)
        np.testing.assert_allclose(inertia, inertia_a, atol=1e-15)

    def test_off_diagonal_products_of_inertia(self):
        """Two point masses separated along (1,1,0): Ixy = -m*L^2/2, Izz = m*L^2.

        The masses lie in the xy-plane, so the out-of-plane axis (z) sees the
        full perpendicular-axis inertia m*L^2, while the in-plane diagonal
        entries are m*L^2/2 with an Ixy = -m*L^2/2 product term.
        """
        m, L = 1.5, 0.4
        mass, com, inertia = combine_inertia(
            m,
            np.zeros(3),
            ZERO6,
            m,
            np.zeros(3),
            ZERO6,
            d=np.array([L, L, 0.0]),
        )
        assert mass == pytest.approx(2 * m)
        np.testing.assert_allclose(com, [L / 2, L / 2, 0], atol=1e-15)
        ixx, iyy, izz, ixy, ixz, iyz = inertia
        np.testing.assert_allclose([ixx, iyy], [m * L**2 / 2] * 2, atol=1e-14)
        assert izz == pytest.approx(m * L**2)
        assert ixy == pytest.approx(-m * L**2 / 2)
        assert ixz == pytest.approx(0.0, abs=1e-15)
        assert iyz == pytest.approx(0.0, abs=1e-15)

    def test_base_inertia_adds_through(self):
        """Isotropic base inertia on each body adds on top of the parallel-axis shift."""
        m, a, i0 = 2.0, 0.3, 0.05
        base = np.array([i0, i0, i0, 0, 0, 0])
        mass, com, inertia = combine_inertia(
            m,
            np.array([-a, 0, 0]),
            base,
            m,
            np.array([a, 0, 0]),
            base,
            d=np.zeros(3),
        )
        np.testing.assert_allclose(
            inertia,
            [2 * i0, 2 * i0 + 2 * m * a**2, 2 * i0 + 2 * m * a**2, 0, 0, 0],
            atol=1e-14,
        )

    def test_zero_total_mass_raises(self):
        with pytest.raises(ValueError):
            combine_inertia(0.0, np.zeros(3), ZERO6, 0.0, np.zeros(3), ZERO6, np.zeros(3))
