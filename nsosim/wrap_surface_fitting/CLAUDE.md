# Wrap Surface Fitting - Context for Claude

This document provides important context about design decisions and implementation details in this module.

## Cylinder Quadrant Orientation Fix

**Commit:** `629e02d` - "Fix cylinder wrap quadrant orientation for consistent fitting"

### Problem

When fitting cylinder wrap surfaces to new bone geometries, the quadrant parameter (which controls which side of a partial cylinder is active for muscle/ligament wrapping in OpenSim) was becoming misaligned. This caused:

1. Cylinders rotated by ~90 degrees from expected orientation
2. The Euler Z-angle (twist around cylinder axis) off by +/- pi/2 radians
3. Muscles/ligaments wrapping on the wrong side of the bone during simulation

### Root Cause

Two issues in the original implementation:

1. **`construct_cylinder_basis()`** chose arbitrary X/Y axes based on a "least aligned" heuristic:
   - Caused inconsistent orientation across different cylinders
   - 90-degree jumps as axis orientation changed slightly
   - No preference for anatomically meaningful directions

2. **`enforce_sign_convention()`** (in `RotationUtils`) flipped rotation matrix columns to canonicalize Euler angles:
   - Changed the twist around the cylinder axis arbitrarily
   - Moved the quadrant to the wrong side
   - Helpful for ellipsoids but harmful for partial cylinders

### Solution

1. **Updated `construct_cylinder_basis()`** (fitting.py:1416):
   - Added `reference_x_axis` parameter (defaults to global +X for anatomical consistency)
   - Added sign consistency check: `if dot(x_local, preferred) < 0: x_local = -x_local`
   - This prevents 180-degree flips that would change the quadrant

2. **Removed `enforce_sign_convention()` for cylinders** in `CylinderFitter.wrap_params`:
   - The rotation from `construct_cylinder_basis` is already deterministic
   - Ellipsoids still use `enforce_sign_convention` (they don't have quadrant sensitivity)

### Why This Matters

Cylinders have rotational symmetry around their long axis. The optimization determines:
- **Axis direction (Z-axis):** Determined by fitting to data
- **Twist around axis (X/Y orientation):** Can be chosen arbitrarily

We now choose the twist to:
1. Align X-axis with global +X when possible (anterior direction for femur)
2. Keep consistent sign (no 180-degree flips)
3. Preserve original OpenSim quadrant semantics

For OpenSim XYZ body-fixed Euler angles:
- First two angles (X, Y) primarily control cylinder axis direction
- Last angle (Z) controls twist, i.e., which quadrant faces which direction

### Key Functions

- `construct_cylinder_basis()` - fitting.py:1416
- `sd_cylinder_with_axis()` - fitting.py:1477
- `CylinderFitter.wrap_params` property - fitting.py (uses rotation directly, no sign convention)
- `EllipsoidFitter.wrap_params` property - fitting.py (still uses `enforce_sign_convention`)

---

## General Module Structure

See `/WRAP_SURFACE_FITTING_REFACTORING.md` at repo root for the overall refactoring plan and module architecture.

### Key Design Decisions

1. **SDF-based optimization:** Uses signed distance functions and PyTorch to fit parametric shapes to classified point clouds

2. **PCA initialization:** Smart initialization based on principal component analysis for faster convergence

3. **Squared-hinge margin loss:** Enforces inside/outside classification with configurable margin

4. **Separate handling for cylinders vs ellipsoids:** Due to cylinder quadrant sensitivity (see above)
