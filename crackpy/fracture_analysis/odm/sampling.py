"""ODM sampling defines fixed fitting geometry, measured displacement
interpolation, and crack-tip rebasing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from crackpy.fracture_analysis._interpolation_cache import (
    InterpolationTarget,
    InterpolatorCache,
    ReusableLinearInterpolator,
)
from crackpy.input.input_data import InputData


@dataclass(frozen=True)
class OptimizationGrid:
    """Describe the fixed polar and Cartesian sampling grid used by ODM fitting.

    Attributes:
        r: Radial coordinate values.
        phi: Angular coordinate values in radians.
        x: Cartesian x-coordinate values.
        y: Cartesian y-coordinate values.
    """

    r: np.ndarray
    phi: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class InterpolatedDisplacementGrid:
    """Describe measured displacements interpolated onto a fixed ODM grid.

    Attributes:
        interpolator: Reusable interpolation geometry for the source and grid.
        tip_displacements: Interpolated displacement components at the crack tip.
        disp_x: Crack-tip-rebased x-displacements on the grid.
        disp_y: Crack-tip-rebased y-displacements on the grid.
        disp_z: Crack-tip-rebased z-displacements on the grid.
    """

    interpolator: ReusableLinearInterpolator
    tip_displacements: np.ndarray
    disp_x: np.ndarray
    disp_y: np.ndarray
    disp_z: np.ndarray


def build_optimization_grid(
    min_radius: float,
    max_radius: float,
    tick_size: float,
    angle_gap_deg: float,
) -> OptimizationGrid:
    """Create the fixed ODM polar grid and corresponding Cartesian coordinates.

    Args:
        min_radius: Inclusive minimum radial coordinate.
        max_radius: Exclusive maximum radial coordinate.
        tick_size: Radial and angular sampling interval.
        angle_gap_deg: Excluded angular gap along the crack path in degrees.

    Returns:
        The fixed ODM optimization grid.
    """
    angle_gap_rad = angle_gap_deg / 180 * np.pi
    r_grid, phi_grid = np.mgrid[
        min_radius:max_radius:tick_size,
        -np.pi + angle_gap_rad:np.pi - angle_gap_rad:tick_size,
    ]
    x_grid = r_grid * np.cos(phi_grid)
    y_grid = r_grid * np.sin(phi_grid)
    return OptimizationGrid(r=r_grid, phi=phi_grid, x=x_grid, y=y_grid)


def build_interpolator_eval_points(grid: OptimizationGrid) -> np.ndarray:
    """Lay out the crack tip followed by flattened ODM grid coordinates.

    Args:
        grid: Fixed ODM optimization grid.

    Returns:
        Evaluation coordinates with the crack tip in the first row.
    """
    return np.r_[np.array([[0.0, 0.0]]), np.c_[grid.x.ravel(), grid.y.ravel()]]


def prepare_interpolated_displacement_grid(
    data: InputData,
    grid: OptimizationGrid,
    interpolator_cache: InterpolatorCache,
) -> InterpolatedDisplacementGrid:
    """Interpolate measured displacements once and rebase them at the crack tip.

    Args:
        data: Crack-tip-centered nodemap data.
        grid: Fixed ODM optimization grid.
        interpolator_cache: Shared interpolation-geometry cache.

    Returns:
        Interpolated and crack-tip-rebased displacement components.
    """
    interpolator = interpolator_cache.get_interpolator(
        data.coor_x,
        data.coor_y,
        build_interpolator_eval_points(grid),
        InterpolationTarget.OPTIMIZATION_GRID,
    )
    raw_displacements = np.c_[data.disp_x, data.disp_y, data.disp_z]
    interpolated_data = interpolator.interpolate(raw_displacements)
    tip_displacements = interpolated_data[0]
    grid_data = interpolated_data[1:] - tip_displacements

    return InterpolatedDisplacementGrid(
        interpolator=interpolator,
        tip_displacements=tip_displacements,
        disp_x=grid_data[:, 0].reshape(grid.x.shape),
        disp_y=grid_data[:, 1].reshape(grid.x.shape),
        disp_z=grid_data[:, 2].reshape(grid.x.shape),
    )
