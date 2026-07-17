from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from enum import Enum
from typing import Generic, TypeVar

import numpy as np
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)

CacheKey = TypeVar("CacheKey")
CacheValue = TypeVar("CacheValue")


class ReusableLinearInterpolator:
    """Precompute barycentric weights for repeated interpolation on fixed points.

    Attributes:
        bary: Barycentric coordinates for each evaluation point.
        vidx: Vertex indices for each evaluation point's simplex.
        valid: Mask identifying evaluation points inside the convex hull.
        n_eval: Number of evaluation points.
    """

    def __init__(
        self,
        coor_x: np.ndarray,
        coor_y: np.ndarray,
        eval_points: np.ndarray,
        tri: Delaunay | None = None,
    ) -> None:
        """Initialize interpolation geometry for scattered source data.

        Args:
            coor_x: One-dimensional x-coordinates at the source points.
            coor_y: One-dimensional y-coordinates at the source points.
            eval_points: Target coordinates with shape ``(M, 2)``.
            tri: Optional precomputed triangulation of the source coordinates.

        Returns:
            None.
        """
        logger.debug(
            "Initializing ReusableLinearInterpolator with %d source points and %d evaluation points",
            len(coor_x),
            len(eval_points),
        )
        if tri is None:
            tri = Delaunay(np.c_[coor_x, coor_y])

        simplex_indices = tri.find_simplex(eval_points)
        transforms = tri.transform[simplex_indices, :2]
        relative_points = eval_points - tri.transform[simplex_indices, 2]
        first_barycentric_coordinates = np.einsum("mij,mj->mi", transforms, relative_points)

        self.bary = np.c_[
            first_barycentric_coordinates,
            1 - first_barycentric_coordinates.sum(axis=1),
        ]
        self.vidx = tri.simplices[simplex_indices]
        self.valid = simplex_indices >= 0
        self.n_eval = eval_points.shape[0]
        logger.debug(
            "Interpolator initialized: %d/%d points inside convex hull",
            np.sum(self.valid),
            self.n_eval,
        )

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """Interpolate source values onto the configured evaluation points.

        Args:
            values: Source values with shape ``(N,)`` or ``(N, k)``.

        Returns:
            Interpolated values with points outside the convex hull set to NaN.
        """
        values = np.asarray(values)
        out_shape = (self.n_eval,) if values.ndim == 1 else (self.n_eval, values.shape[1])
        interpolated = np.full(out_shape, np.nan, dtype=values.dtype)

        if values.ndim == 1:
            interpolated[self.valid] = (
                self.bary[self.valid] * values[self.vidx[self.valid]]
            ).sum(axis=1)
        else:
            interpolated[self.valid] = (
                self.bary[self.valid, :, None] * values[self.vidx[self.valid]]
            ).sum(axis=1)
        return interpolated


class InterpolationTarget(Enum):
    """Identify the intended evaluation layout in interpolation cache keys."""

    INTEGRATION_POINTS = "integration_points"
    INTEGRATION_POINTS_ALL = "integration_points_all"
    OPTIMIZATION_GRID = "optimization_grid"
    REFERENCE_POINT = "reference_point"
    REGULAR_GRID = "regular_grid"


def hash_array(values: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    """Build a content-based identity for a NumPy array.

    Args:
        values: Array whose shape, dtype, and bytes identify a cache input.

    Returns:
        A tuple containing the contiguous shape, dtype string, and BLAKE2 digest.
    """
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()
    return contiguous.shape, contiguous.dtype.str, digest


class BoundedCache(Generic[CacheKey, CacheValue]):
    """Store values in access order with optional least-recently-used eviction.

    Attributes:
        max_size: Maximum number of retained values, or ``None`` for no limit.
    """

    def __init__(self, max_size: int | None = None) -> None:
        """Initialize an empty ordered cache.

        Args:
            max_size: Maximum number of values to retain, or ``None`` for no limit.

        Returns:
            None.
        """
        self.max_size = max_size
        self._store: OrderedDict[CacheKey, CacheValue] = OrderedDict()

    def get(self, key: CacheKey) -> CacheValue | None:
        """Return and mark a cached value as recently used.

        Args:
            key: Cache key to look up.

        Returns:
            The cached value, or ``None`` when the key is absent.
        """
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        return value

    def set(self, key: CacheKey, value: CacheValue) -> None:
        """Store a value and evict the least-recently-used entry if necessary.

        Args:
            key: Cache key for the value.
            value: Value to retain.

        Returns:
            None.
        """
        self._store[key] = value
        self._store.move_to_end(key)
        if self.max_size is not None and len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all cached values.

        Returns:
            None.
        """
        self._store.clear()

class InterpolatorCache:
    """Reuse triangulations and interpolation geometry across instances."""

    def __init__(self, max_interpolators: int | None = None) -> None:
        """Initialize the shared geometry caches.

        Args:
            max_interpolators: Maximum number of interpolators to retain.

        Returns:
            None.
        """
        self._source_triangles: BoundedCache[tuple, Delaunay] = BoundedCache(max_interpolators)
        self._interpolators: BoundedCache[tuple, ReusableLinearInterpolator] = BoundedCache(max_interpolators)

    def source_key(self, coor_x: np.ndarray, coor_y: np.ndarray) -> tuple:
        """Create a content-based key for source coordinates.

        Args:
            coor_x: Source x-coordinates.
            coor_y: Source y-coordinates.

        Returns:
            A cache key identifying the source geometry.
        """
        return hash_array(coor_x), hash_array(coor_y)

    def interpolator_key(
        self,
        coor_x: np.ndarray,
        coor_y: np.ndarray,
        eval_points: np.ndarray,
        target: InterpolationTarget,
    ) -> tuple:
        """Create a content-based key for source and target geometry.

        Args:
            coor_x: Source x-coordinates.
            coor_y: Source y-coordinates.
            eval_points: Interpolation evaluation coordinates.
            target: Semantic evaluation layout.

        Returns:
            A cache key identifying the complete interpolation geometry.
        """
        return self.source_key(coor_x, coor_y) + (hash_array(eval_points), target)

    def get_interpolator(
        self,
        coor_x: np.ndarray,
        coor_y: np.ndarray,
        eval_points: np.ndarray,
        target: InterpolationTarget,
    ) -> ReusableLinearInterpolator:
        """Return reusable interpolation geometry, constructing it when absent.

        Args:
            coor_x: Source x-coordinates.
            coor_y: Source y-coordinates.
            eval_points: Interpolation evaluation coordinates.
            target: Semantic evaluation layout.

        Returns:
            The cached or newly constructed interpolator.
        """
        cache_key = self.interpolator_key(coor_x, coor_y, eval_points, target)
        interpolator = self._interpolators.get(cache_key)
        if interpolator is None:
            source_key = cache_key[:2]
            tri = self._source_triangles.get(source_key)
            if tri is None:
                tri = Delaunay(np.c_[coor_x, coor_y])
                self._source_triangles.set(source_key, tri)
            interpolator = ReusableLinearInterpolator(coor_x, coor_y, eval_points, tri=tri)
            self._interpolators.set(cache_key, interpolator)
        return interpolator

    def clear(self) -> None:
        """Clear all cached triangulations and interpolators.

        Returns:
            None.
        """
        self._source_triangles.clear()
        self._interpolators.clear()
