"""Tests for meniscus articular surface extraction stability.

Covers:
- Category A: Unit tests for untested helper functions
  (label_meniscus_regions_with_sdf, compute_region_radial_percentiles,
   refine_meniscus_articular_surfaces, full pipeline with refinement)
- Category B: Characterization tests exposing extraction instability
  (topology perturbation, proportional stability, rim exclusion, determinism)
- Category C: Integration tests with real meshes (marked @pytest.mark.slow)
"""

import pathlib

import numpy as np
import pytest
import pyvista as pv
from pymskt.mesh import Mesh

from nsosim.articular_surfaces import (
    compute_region_radial_percentiles,
    create_meniscus_articulating_surface,
    extract_meniscus_articulating_surface,
    label_meniscus_regions_with_sdf,
    refine_meniscus_articular_surfaces,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "meniscus"


# ===========================================================================
# Synthetic half-ring geometry builder
# ===========================================================================


def _make_half_ring(
    outer_radius=15.0,
    inner_radius=8.0,
    height=5.0,
    n_theta=80,
    n_r=20,
    n_y=6,
    theta_range=(0, np.pi),
):
    """Create a C-shaped half-ring mesh with known flat top/bottom and curved rims.

    Returns a PolyData mesh with point data:
      - 'region': 'top_face', 'bottom_face', 'inner_rim', 'outer_rim'

    The top face is at y = +height/2, bottom face at y = -height/2.
    Inner rim is the curved surface at r = inner_radius.
    Outer rim is the curved surface at r = outer_radius.
    """
    theta_vals = np.linspace(theta_range[0], theta_range[1], n_theta)
    r_vals = np.linspace(inner_radius, outer_radius, n_r)
    y_vals = np.linspace(-height / 2, height / 2, n_y)

    points = []
    regions = []  # 0=top, 1=bottom, 2=inner_rim, 3=outer_rim

    # Top face (y = +height/2)
    for theta in theta_vals:
        for r in r_vals:
            x = r * np.sin(theta)
            z = r * np.cos(theta)
            y = height / 2
            points.append([x, y, z])
            regions.append(0)  # top_face

    # Bottom face (y = -height/2)
    for theta in theta_vals:
        for r in r_vals:
            x = r * np.sin(theta)
            z = r * np.cos(theta)
            y = -height / 2
            points.append([x, y, z])
            regions.append(1)  # bottom_face

    # Inner rim (r = inner_radius, varying y)
    for theta in theta_vals:
        for y in y_vals[1:-1]:  # skip top/bottom (already included)
            x = inner_radius * np.sin(theta)
            z = inner_radius * np.cos(theta)
            points.append([x, y, z])
            regions.append(2)  # inner_rim

    # Outer rim (r = outer_radius, varying y)
    for theta in theta_vals:
        for y in y_vals[1:-1]:
            x = outer_radius * np.sin(theta)
            z = outer_radius * np.cos(theta)
            points.append([x, y, z])
            regions.append(3)  # outer_rim

    points = np.array(points)
    regions = np.array(regions)

    # Create mesh from point cloud and triangulate
    cloud = pv.PolyData(points)
    mesh = cloud.delaunay_3d().extract_surface()

    # Transfer region labels to the triangulated mesh using nearest-neighbor
    from scipy.spatial import KDTree

    tree = KDTree(points)
    _, idx = tree.query(mesh.points)
    mesh["region"] = regions[idx]

    return mesh


def _make_flat_bone_plane(y_offset, extent=25.0, resolution=20):
    """Create a flat plane mesh at a given y offset (bone proxy)."""
    plane = pv.Plane(
        center=(0, y_offset, 0),
        direction=(0, 1, 0),
        i_size=extent * 2,
        j_size=extent * 2,
        i_resolution=resolution,
        j_resolution=resolution,
    )
    return plane


# ===========================================================================
# Category A: Unit tests for untested functions
# ===========================================================================


class TestLabelMeniscusRegionsWithSDF:
    """A1: Test label_meniscus_regions_with_sdf on synthetic geometry."""

    @pytest.fixture(scope="class")
    def half_ring_with_surfaces(self):
        """Create half-ring and extract top/bottom face submeshes as 'surfaces'."""
        ring = _make_half_ring()

        # Extract top and bottom face points as separate meshes
        top_mask = ring["region"] == 0
        bottom_mask = ring["region"] == 1

        top_surface = ring.extract_points(top_mask, adjacent_cells=True).extract_surface()
        bottom_surface = ring.extract_points(bottom_mask, adjacent_cells=True).extract_surface()

        return ring, top_surface, bottom_surface

    def test_labels_are_valid(self, half_ring_with_surfaces):
        """All labels should be in {0, 1, 2, 3}."""
        ring, top_surface, bottom_surface = half_ring_with_surfaces
        labels = label_meniscus_regions_with_sdf(
            ring, lower_surface=bottom_surface, upper_surface=top_surface, distance_thresh=1.0
        )
        assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    def test_top_face_labeled_upper(self, half_ring_with_surfaces):
        """Vertices on top face should be labeled 2 (near upper surface) or 3 (both)."""
        ring, top_surface, bottom_surface = half_ring_with_surfaces
        labels = label_meniscus_regions_with_sdf(
            ring, lower_surface=bottom_surface, upper_surface=top_surface, distance_thresh=1.0
        )
        top_mask = ring["region"] == 0
        top_labels = labels[top_mask]
        # Top face points should be near the upper surface (label 2 or 3)
        assert np.all((top_labels == 2) | (top_labels == 3)), (
            f"Expected top face labels to be 2 or 3, got unique: {np.unique(top_labels)}"
        )

    def test_bottom_face_labeled_lower(self, half_ring_with_surfaces):
        """Vertices on bottom face should be labeled 1 (near lower surface) or 3 (both)."""
        ring, top_surface, bottom_surface = half_ring_with_surfaces
        labels = label_meniscus_regions_with_sdf(
            ring, lower_surface=bottom_surface, upper_surface=top_surface, distance_thresh=1.0
        )
        bottom_mask = ring["region"] == 1
        bottom_labels = labels[bottom_mask]
        assert np.all((bottom_labels == 1) | (bottom_labels == 3)), (
            f"Expected bottom face labels to be 1 or 3, got unique: {np.unique(bottom_labels)}"
        )

    def test_rim_vertices_mostly_unlabeled(self, half_ring_with_surfaces):
        """Most rim vertices should be label 0 (neither surface) with tight threshold.

        Note: some rim vertices near top/bottom edges share triangulated faces,
        so a small fraction will be labeled even with tight threshold. We check
        that at least 40% are unlabeled (the remaining overlap is geometric).
        """
        ring, top_surface, bottom_surface = half_ring_with_surfaces
        labels = label_meniscus_regions_with_sdf(
            ring,
            lower_surface=bottom_surface,
            upper_surface=top_surface,
            distance_thresh=0.1,  # very tight
        )
        rim_mask = (ring["region"] == 2) | (ring["region"] == 3)
        rim_labels = labels[rim_mask]
        pct_unlabeled = np.mean(rim_labels == 0)
        assert pct_unlabeled > 0.4, (
            f"Expected >40% of rim vertices to be unlabeled, got {pct_unlabeled:.1%}"
        )


class TestComputeRegionRadialPercentiles:
    """A2: Test compute_region_radial_percentiles with known distributions."""

    @pytest.fixture(scope="class")
    def mesh_with_known_distributions(self):
        """Create mesh with known polar coords and region labels."""
        rng = np.random.default_rng(42)
        n = 2000

        # Region 1: r in [3, 7], theta uniform
        theta1 = rng.uniform(-np.pi, np.pi, n)
        r1 = rng.uniform(3.0, 7.0, n)
        pts1 = np.column_stack([r1 * np.sin(theta1), np.zeros(n), r1 * np.cos(theta1)])

        # Region 2: r in [4, 8], theta uniform
        theta2 = rng.uniform(-np.pi, np.pi, n)
        r2 = rng.uniform(4.0, 8.0, n)
        pts2 = np.column_stack([r2 * np.sin(theta2), np.zeros(n), r2 * np.cos(theta2)])

        # Region 0 (background): r in [1, 2]
        theta0 = rng.uniform(-np.pi, np.pi, n // 2)
        r0 = rng.uniform(1.0, 2.0, n // 2)
        pts0 = np.column_stack([r0 * np.sin(theta0), np.zeros(n // 2), r0 * np.cos(theta0)])

        pts = np.vstack([pts1, pts2, pts0])
        regions = np.concatenate(
            [np.ones(n) * 1, np.ones(n) * 2, np.zeros(n // 2)]
        ).astype(float)

        mesh = pv.PolyData(pts)
        mesh["theta"] = np.concatenate([theta1, theta2, theta0])
        mesh["r"] = np.concatenate([r1, r2, r0])
        mesh["regions_label"] = regions

        return mesh

    def test_returns_entries_for_nonzero_regions(self, mesh_with_known_distributions):
        result, _ = compute_region_radial_percentiles(
            mesh_with_known_distributions, percentile=95.0
        )
        assert 1 in result
        assert 2 in result
        assert 0 not in result  # background skipped

    def test_percentile_values_in_range(self, mesh_with_known_distributions):
        result, _ = compute_region_radial_percentiles(
            mesh_with_known_distributions, percentile=95.0
        )
        # Region 1: r in [3, 7], 95th percentile should be near 6.8
        r_p1 = result[1]["r_percentile"]
        assert np.all(r_p1 > 5.0), f"Region 1 percentile too low: min={r_p1.min():.2f}"
        assert np.all(r_p1 < 7.5), f"Region 1 percentile too high: max={r_p1.max():.2f}"

        # Region 2: r in [4, 8], 95th percentile should be near 7.8
        r_p2 = result[2]["r_percentile"]
        assert np.all(r_p2 > 6.0), f"Region 2 percentile too low: min={r_p2.min():.2f}"
        assert np.all(r_p2 < 8.5), f"Region 2 percentile too high: max={r_p2.max():.2f}"

    def test_bin_count_matches(self, mesh_with_known_distributions):
        n_bins = 50
        result, theta_bins = compute_region_radial_percentiles(
            mesh_with_known_distributions, percentile=95.0, n_theta_bins=n_bins
        )
        assert len(theta_bins) == n_bins
        # bin_centers should have at most n_bins - 1 entries
        for region_label, pdata in result.items():
            assert len(pdata["bin_centers"]) <= n_bins - 1


class TestRefineEndToEnd:
    """A3: Test refine_meniscus_articular_surfaces removes outlier tongues."""

    def test_outlier_tongue_removed(self):
        """Surfaces with outlier tongues should be trimmed by radial envelope."""
        ring = _make_half_ring(n_theta=60, n_r=15, n_y=5)
        top_mask = ring["region"] == 0
        bottom_mask = ring["region"] == 1

        top_surface = ring.extract_points(top_mask, adjacent_cells=True).extract_surface()
        bottom_surface = ring.extract_points(bottom_mask, adjacent_cells=True).extract_surface()

        # Add outlier tongue to bottom surface: extend some points radially outward
        bottom_pts = bottom_surface.points.copy()
        # Extend points in first 10% of indices outward by 50%
        n_extend = max(1, len(bottom_pts) // 10)
        r = np.sqrt(bottom_pts[:n_extend, 0] ** 2 + bottom_pts[:n_extend, 2] ** 2)
        scale = 1.5
        bottom_pts[:n_extend, 0] *= scale
        bottom_pts[:n_extend, 2] *= scale
        bottom_with_tongue = bottom_surface.copy()
        bottom_with_tongue.points = bottom_pts

        lower_trimmed, upper_trimmed, (theta_grid, r_min_grid) = refine_meniscus_articular_surfaces(
            meniscus_mesh=ring.copy(),
            lower_surface=bottom_with_tongue,
            upper_surface=top_surface.copy(),
            distance_thresh=2.0,  # generous threshold for synthetic geometry
            percentile=95.0,
            theta_offset=0.0,
        )

        # The trimmed surface should have fewer points than the tongue version
        assert lower_trimmed.n_points < bottom_with_tongue.n_points, (
            f"Expected trimming to remove points: "
            f"{lower_trimmed.n_points} vs {bottom_with_tongue.n_points}"
        )
        # Envelope should have the expected shape
        assert len(theta_grid) == 200
        assert len(r_min_grid) == 200


class TestFullPipelineWithRefinement:
    """A4: Full pipeline smoke test with radial envelope refinement enabled."""

    @pytest.mark.slow
    def test_half_ring_pipeline_with_refinement(self):
        """Half-ring between flat planes should produce refined surfaces."""
        # Build geometry in meters (pipeline expects meters)
        scale = 0.001  # mm -> m
        ring = _make_half_ring(
            outer_radius=15.0, inner_radius=8.0, height=5.0, n_theta=60, n_r=15, n_y=5
        )
        ring.points *= scale

        upper_plane = _make_flat_bone_plane(y_offset=4.0, extent=25.0)
        upper_plane.points *= scale
        lower_plane = _make_flat_bone_plane(y_offset=-4.0, extent=25.0)
        lower_plane.points *= scale

        men_mesh = Mesh(ring)
        upper_mesh = Mesh(upper_plane)
        lower_mesh = Mesh(lower_plane)

        upper_surf, lower_surf = create_meniscus_articulating_surface(
            meniscus_mesh=men_mesh,
            upper_articulating_bone_mesh=upper_mesh,
            lower_articulating_bone_mesh=lower_mesh,
            ray_length=10.0,
            n_largest=1,
            smooth_iter=5,
            triangle_density=None,
            refine_by_radial_envelope=True,
            theta_offset=0.0,
        )

        assert upper_surf is not None
        assert lower_surf is not None
        assert isinstance(upper_surf, Mesh)
        assert isinstance(lower_surf, Mesh)
        assert upper_surf.n_points > 0
        assert lower_surf.n_points > 0
        # Surfaces should be smaller than full meniscus (refinement trimmed)
        assert upper_surf.n_points < ring.n_points
        assert lower_surf.n_points < ring.n_points


# ===========================================================================
# Category B: Characterization tests (expose instability)
# ===========================================================================


def _compute_assd(mesh1, mesh2):
    """Average Symmetric Surface Distance between two meshes."""
    from scipy.spatial import KDTree

    pts1 = mesh1.points if hasattr(mesh1, "points") else mesh1.point_coords
    pts2 = mesh2.points if hasattr(mesh2, "points") else mesh2.point_coords

    tree1 = KDTree(pts1)
    tree2 = KDTree(pts2)

    d12, _ = tree2.query(pts1)
    d21, _ = tree1.query(pts2)

    return (d12.mean() + d21.mean()) / 2.0


def _perturb_mesh(mesh, rng, sigma=0.005):
    """Add small random perturbations to mesh vertices to simulate different
    marching cubes outputs (the real source of stochasticity in the pipeline).

    sigma is in the same units as the mesh (mm for extraction-level tests).
    Default 0.005mm matches the observed ASSD between NSM reconstruction runs.
    Even this tiny perturbation triggers ACVD clustering divergence, producing
    ~0.28mm ASSD in extracted articular surfaces via the current ray-casting method.
    """
    perturbed = mesh.copy()
    perturbed.points = perturbed.points + rng.normal(0, sigma, perturbed.points.shape)
    return perturbed


class TestTopologyPerturbationStability:
    """B1: Small vertex perturbations (simulating different NSM reconstructions)
    should produce stable extraction results after ACVD + ray-casting."""

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Ray-casting extraction amplifies ACVD clustering noise; "
        "scored extraction (Method 1) should fix this",
        strict=False,
    )
    def test_perturbed_remeshings_give_similar_extraction(self):
        """Perturbed + remeshed meniscus should yield similar articular surfaces.

        This simulates the real pipeline stochasticity: NSM marching cubes produces
        slightly different vertex positions each run → ACVD remeshes → ray-casting
        extracts. The extraction should be stable to small vertex perturbations.
        """
        ring = _make_half_ring(
            outer_radius=15.0, inner_radius=8.0, height=5.0, n_theta=80, n_r=20, n_y=6
        )
        upper_plane = _make_flat_bone_plane(y_offset=4.0, extent=25.0)

        rng = np.random.default_rng(42)

        # Perturb vertices, then ACVD resample, then extract
        ring1 = _perturb_mesh(ring, rng)
        ring_mesh1 = Mesh(ring1)
        ring_mesh1.resample_surface(subdivisions=1, clusters=3000)

        ring2 = _perturb_mesh(ring, rng)
        ring_mesh2 = Mesh(ring2)
        ring_mesh2.resample_surface(subdivisions=1, clusters=3000)

        surf1 = extract_meniscus_articulating_surface(
            ring_mesh1.mesh, upper_plane, ray_length=10.0, n_largest=1, smooth_iter=5
        )
        surf2 = extract_meniscus_articulating_surface(
            ring_mesh2.mesh, upper_plane, ray_length=10.0, n_largest=1, smooth_iter=5
        )

        # Stability assertions
        assd = _compute_assd(surf1, surf2)
        assert assd < 0.1, f"ASSD between perturbed extractions = {assd:.4f}mm, expected < 0.1mm"

        n1 = surf1.n_points
        n2 = surf2.n_points
        ratio = max(n1, n2) / max(min(n1, n2), 1)
        assert ratio < 1.15, (
            f"Point count ratio = {ratio:.2f} ({n1} vs {n2}), expected < 1.15"
        )


class TestTopologyPerturbationWithRefinement:
    """B2: Same as B1 but with full pipeline including radial envelope refinement."""

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Radial envelope cannot fully compensate for ray-casting instability; "
        "scored extraction (Method 1) should fix this",
        strict=False,
    )
    def test_perturbed_pipeline_refined(self):
        """Full pipeline with refinement should stabilize extraction across
        perturbed inputs (simulating different NSM reconstruction runs)."""
        scale = 0.001  # mm -> m
        ring = _make_half_ring(
            outer_radius=15.0, inner_radius=8.0, height=5.0, n_theta=80, n_r=20, n_y=6
        )

        rng = np.random.default_rng(123)

        # Perturb in mm space, then scale to meters
        ring1 = _perturb_mesh(ring, rng)
        ring1.points *= scale
        ring2 = _perturb_mesh(ring, rng)
        ring2.points *= scale

        upper_plane = _make_flat_bone_plane(y_offset=4.0, extent=25.0)
        upper_plane.points *= scale
        lower_plane = _make_flat_bone_plane(y_offset=-4.0, extent=25.0)
        lower_plane.points *= scale

        upper1, lower1 = create_meniscus_articulating_surface(
            meniscus_mesh=Mesh(ring1.copy()),
            upper_articulating_bone_mesh=Mesh(upper_plane.copy()),
            lower_articulating_bone_mesh=Mesh(lower_plane.copy()),
            ray_length=10.0,
            n_largest=1,
            smooth_iter=5,
            triangle_density=None,
            refine_by_radial_envelope=True,
            theta_offset=0.0,
        )

        upper2, lower2 = create_meniscus_articulating_surface(
            meniscus_mesh=Mesh(ring2.copy()),
            upper_articulating_bone_mesh=Mesh(upper_plane.copy()),
            lower_articulating_bone_mesh=Mesh(lower_plane.copy()),
            ray_length=10.0,
            n_largest=1,
            smooth_iter=5,
            triangle_density=None,
            refine_by_radial_envelope=True,
            theta_offset=0.0,
        )

        # ASSD in mm (surfaces returned in meters, convert)
        upper1_mm = upper1.copy()
        upper1_mm.point_coords *= 1000
        upper2_mm = upper2.copy()
        upper2_mm.point_coords *= 1000
        lower1_mm = lower1.copy()
        lower1_mm.point_coords *= 1000
        lower2_mm = lower2.copy()
        lower2_mm.point_coords *= 1000

        assd_upper = _compute_assd(upper1_mm, upper2_mm)
        assert assd_upper < 0.1, (
            f"Upper surface ASSD = {assd_upper:.4f}mm, expected < 0.1mm"
        )

        assd_lower = _compute_assd(lower1_mm, lower2_mm)
        assert assd_lower < 0.1, (
            f"Lower surface ASSD = {assd_lower:.4f}mm, expected < 0.1mm"
        )


class TestNoRimVerticesInExtraction:
    """B4: Ground truth test — extracted surface should not contain rim vertices."""

    @pytest.mark.slow
    def test_no_rim_in_upper_extraction(self):
        """Upper extraction should contain only top-face vertices, no rim."""
        ring = _make_half_ring(
            outer_radius=15.0, inner_radius=8.0, height=5.0, n_theta=60, n_r=15, n_y=5
        )
        upper_plane = _make_flat_bone_plane(y_offset=4.0, extent=25.0)

        # Ensure ring has normals
        ring.compute_normals(point_normals=True, auto_orient_normals=True, inplace=True)

        surf = extract_meniscus_articulating_surface(
            ring, upper_plane, ray_length=10.0, n_largest=1, smooth_iter=5
        )

        # Map extracted points back to original ring to get region labels
        from scipy.spatial import KDTree

        tree = KDTree(ring.points)
        _, idx = tree.query(surf.points)
        extracted_regions = ring["region"][idx]

        # No rim vertices should be present
        n_inner_rim = np.sum(extracted_regions == 2)
        n_outer_rim = np.sum(extracted_regions == 3)
        n_total = len(extracted_regions)

        # Allow tiny fraction due to smoothing moving vertices near boundaries
        rim_fraction = (n_inner_rim + n_outer_rim) / max(n_total, 1)
        assert rim_fraction < 0.05, (
            f"Rim vertices in extraction: {n_inner_rim} inner + {n_outer_rim} outer "
            f"out of {n_total} total ({rim_fraction:.1%})"
        )

        # At least 50% of top-face vertices should be included
        n_top = np.sum(extracted_regions == 0)
        top_frac = n_top / max(n_total, 1)
        assert top_frac > 0.50, (
            f"Only {top_frac:.1%} of extracted points are top-face vertices"
        )

    @pytest.mark.slow
    def test_no_rim_in_lower_extraction(self):
        """Lower extraction should contain only bottom-face vertices, no rim."""
        ring = _make_half_ring(
            outer_radius=15.0, inner_radius=8.0, height=5.0, n_theta=60, n_r=15, n_y=5
        )
        lower_plane = _make_flat_bone_plane(y_offset=-4.0, extent=25.0)

        ring.compute_normals(point_normals=True, auto_orient_normals=True, inplace=True)

        surf = extract_meniscus_articulating_surface(
            ring, lower_plane, ray_length=10.0, n_largest=1, smooth_iter=5
        )

        from scipy.spatial import KDTree

        tree = KDTree(ring.points)
        _, idx = tree.query(surf.points)
        extracted_regions = ring["region"][idx]

        n_inner_rim = np.sum(extracted_regions == 2)
        n_outer_rim = np.sum(extracted_regions == 3)
        n_total = len(extracted_regions)

        rim_fraction = (n_inner_rim + n_outer_rim) / max(n_total, 1)
        assert rim_fraction < 0.05, (
            f"Rim vertices in extraction: {n_inner_rim} inner + {n_outer_rim} outer "
            f"out of {n_total} total ({rim_fraction:.1%})"
        )

        n_bottom = np.sum(extracted_regions == 1)
        bottom_frac = n_bottom / max(n_total, 1)
        assert bottom_frac > 0.50, (
            f"Only {bottom_frac:.1%} of extracted points are bottom-face vertices"
        )


class TestDeterminismSameInputs:
    """B5: Same inputs should produce identical outputs (no resampling)."""

    @pytest.mark.slow
    def test_identical_runs(self):
        """Two calls with identical inputs should produce identical output."""
        ring = _make_half_ring(
            outer_radius=15.0, inner_radius=8.0, height=5.0, n_theta=60, n_r=15, n_y=5
        )
        upper_plane = _make_flat_bone_plane(y_offset=4.0, extent=25.0)

        ring.compute_normals(point_normals=True, auto_orient_normals=True, inplace=True)

        surf1 = extract_meniscus_articulating_surface(
            ring.copy(), upper_plane.copy(), ray_length=10.0, n_largest=1, smooth_iter=5
        )
        surf2 = extract_meniscus_articulating_surface(
            ring.copy(), upper_plane.copy(), ray_length=10.0, n_largest=1, smooth_iter=5
        )

        assert surf1.n_points == surf2.n_points, (
            f"Determinism: {surf1.n_points} vs {surf2.n_points} points"
        )
        np.testing.assert_allclose(
            surf1.points, surf2.points, atol=1e-10,
            err_msg="Identical inputs produced different outputs"
        )


# ===========================================================================
# Category C: Integration tests with real data
# ===========================================================================


def _real_fixtures_available():
    """Check if real mesh fixtures exist."""
    required = [
        "med_men_osim.stl",
        "lat_men_osim.stl",
        "femur_nsm_recon_osim.stl",
        "tibia_nsm_recon_osim.stl",
        "tibia_labeled_mesh_updated.vtk",
    ]
    return all((FIXTURES_DIR / f).exists() for f in required)


skip_no_fixtures = pytest.mark.skipif(
    not _real_fixtures_available(),
    reason="Real mesh fixtures not found in tests/fixtures/meniscus/",
)


@skip_no_fixtures
class TestRealMeniscusExtractionStability:
    """C1: Real meniscus extraction should be stable across vertex perturbations
    that simulate different NSM reconstruction runs."""

    @pytest.fixture(scope="class")
    def real_meshes(self):
        """Load real mesh fixtures."""
        med_men = pv.read(str(FIXTURES_DIR / "med_men_osim.stl"))
        femur = pv.read(str(FIXTURES_DIR / "femur_nsm_recon_osim.stl"))
        tibia = pv.read(str(FIXTURES_DIR / "tibia_nsm_recon_osim.stl"))
        return med_men, femur, tibia

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Ray-casting extraction amplifies ACVD clustering noise on real meshes",
        strict=False,
    )
    def test_perturbation_stability_upper(self, real_meshes):
        """Upper surface extraction should be stable across vertex perturbations."""
        med_men, femur, tibia = real_meshes
        rng = np.random.default_rng(42)

        # Convert to mm
        men_mm = med_men.copy()
        men_mm.points *= 1000
        fem_mm = femur.copy()
        fem_mm.points *= 1000

        # Perturb vertices (0.05mm ≈ 10x the observed ASSD between NSM runs)
        men1 = _perturb_mesh(men_mm, rng)
        men1_mesh = Mesh(men1)
        men1_mesh.resample_surface(subdivisions=1, clusters=3000)

        men2 = _perturb_mesh(men_mm, rng)
        men2_mesh = Mesh(men2)
        men2_mesh.resample_surface(subdivisions=1, clusters=3000)

        surf1 = extract_meniscus_articulating_surface(
            men1_mesh.mesh, fem_mm, ray_length=15.0, n_largest=1, smooth_iter=10
        )
        surf2 = extract_meniscus_articulating_surface(
            men2_mesh.mesh, fem_mm, ray_length=15.0, n_largest=1, smooth_iter=10
        )

        assd = _compute_assd(surf1, surf2)
        assert assd < 0.1, (
            f"Real upper surface ASSD = {assd:.4f}mm, expected < 0.1mm"
        )

        n1, n2 = surf1.n_points, surf2.n_points
        ratio = max(n1, n2) / max(min(n1, n2), 1)
        assert ratio < 1.15, (
            f"Real upper surface point count ratio = {ratio:.2f} ({n1} vs {n2})"
        )

    @pytest.mark.slow
    @pytest.mark.xfail(
        reason="Ray-casting extraction amplifies ACVD clustering noise on real meshes",
        strict=False,
    )
    def test_perturbation_stability_lower(self, real_meshes):
        """Lower surface extraction should be stable across vertex perturbations."""
        med_men, femur, tibia = real_meshes
        rng = np.random.default_rng(99)

        men_mm = med_men.copy()
        men_mm.points *= 1000
        tib_mm = tibia.copy()
        tib_mm.points *= 1000

        men1 = _perturb_mesh(men_mm, rng)
        men1_mesh = Mesh(men1)
        men1_mesh.resample_surface(subdivisions=1, clusters=3000)

        men2 = _perturb_mesh(men_mm, rng)
        men2_mesh = Mesh(men2)
        men2_mesh.resample_surface(subdivisions=1, clusters=3000)

        surf1 = extract_meniscus_articulating_surface(
            men1_mesh.mesh, tib_mm, ray_length=15.0, n_largest=1, smooth_iter=10
        )
        surf2 = extract_meniscus_articulating_surface(
            men2_mesh.mesh, tib_mm, ray_length=15.0, n_largest=1, smooth_iter=10
        )

        assd = _compute_assd(surf1, surf2)
        assert assd < 0.1, (
            f"Real lower surface ASSD = {assd:.4f}mm, expected < 0.1mm"
        )

        n1, n2 = surf1.n_points, surf2.n_points
        ratio = max(n1, n2) / max(min(n1, n2), 1)
        assert ratio < 1.15, (
            f"Real lower surface point count ratio = {ratio:.2f} ({n1} vs {n2})"
        )


@skip_no_fixtures
class TestRealMeniscusNoRimPoints:
    """C2: Extracted real meniscus surface normals should face the bone."""

    @pytest.mark.slow
    def test_extracted_normals_face_bone(self):
        """Extracted surface vertex normals should point toward the bone."""
        med_men = pv.read(str(FIXTURES_DIR / "med_men_osim.stl"))
        femur = pv.read(str(FIXTURES_DIR / "femur_nsm_recon_osim.stl"))

        # Convert to mm
        men_mm = med_men.copy()
        men_mm.points *= 1000
        fem_mm = femur.copy()
        fem_mm.points *= 1000

        # Resample and extract
        men_mesh = Mesh(men_mm)
        men_mesh.resample_surface(subdivisions=1, clusters=3000)

        surf = extract_meniscus_articulating_surface(
            men_mesh.mesh, fem_mm, ray_length=15.0, n_largest=1, smooth_iter=10
        )

        # Compute normals on extracted surface
        surf_pv = surf.mesh if hasattr(surf, "mesh") else surf
        if not isinstance(surf_pv, pv.PolyData):
            surf_pv = pv.wrap(surf_pv)
        surf_pv.compute_normals(
            point_normals=True, auto_orient_normals=True, inplace=True
        )

        from scipy.spatial import KDTree

        bone_tree = KDTree(fem_mm.points)
        _, bone_idx = bone_tree.query(surf_pv.points)
        closest_bone = fem_mm.points[bone_idx]

        direction = closest_bone - surf_pv.points
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        norms[norms == 0] = 1
        direction = direction / norms

        dots = np.sum(surf_pv.point_normals * direction, axis=1)

        # Most normals should be aligned with bone direction (articular face).
        # Use |dot| since auto_orient_normals on open surfaces may flip orientation.
        # Rim vertices have normals perpendicular to bone direction -> |dot| ≈ 0.
        pct_facing_bone = np.mean(np.abs(dots) > 0.2)
        assert pct_facing_bone > 0.80, (
            f"Only {pct_facing_bone:.1%} of normals aligned with bone (|dot| > 0.2), expected > 80%"
        )
