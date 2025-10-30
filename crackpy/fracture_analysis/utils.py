import logging

import numpy as np
from scipy.spatial import Delaunay

logger = logging.getLogger(__name__)


class ReusableLinearInterpolator:
    """Precompute Delaunay triangulation and barycentric weights for a fixed set of
    evaluation points (grid or arbitrary points). Reuse for many fields efficiently.

    This class significantly improves performance when interpolating multiple fields
    on the same grid by precomputing the triangulation and weights.

    Attributes:
        bary: Barycentric coordinates for each evaluation point
        vidx: Vertex indices for each simplex
        valid: Mask indicating which points are inside the convex hull
        n_eval: Number of evaluation points

    """

    def __init__(self, coor_x: np.ndarray, coor_y: np.ndarray, eval_points: np.ndarray) -> None:
        """Initialize interpolator with scattered data points and evaluation points.

        Args:
            coor_x: 1D array of x-coordinates at measurement points (length N)
            coor_y: 1D array of y-coordinates at measurement points (length N)
            eval_points: (M, 2) array of target points where interpolation is desired

        """
        # use lazy logging to avoid eager string formatting
        logger.debug("Initializing ReusableLinearInterpolator with %d source points and %d evaluation points",
                     len(coor_x), len(eval_points))

        pts = np.c_[coor_x, coor_y]  # (N, 2)
        tri = Delaunay(pts)

        simp = tri.find_simplex(eval_points)  # simplex index per eval point
        T = tri.transform[simp, :2]  # affine transforms
        R = eval_points - tri.transform[simp, 2]  # shifted eval points
        bary12 = np.einsum('mij,mj->mi', T, R)  # first 2 barycentric coords

        self.bary: np.ndarray = np.c_[bary12, 1 - bary12.sum(axis=1)]  # full barycentric weights
        self.vidx: np.ndarray = tri.simplices[simp]  # vertex indices
        self.valid: np.ndarray = simp >= 0  # inside convex hull mask
        self.n_eval: int = eval_points.shape[0]

        n_valid = np.sum(self.valid)
        logger.debug("Interpolator initialized: %d/%d points inside convex hull", n_valid, self.n_eval)

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """Interpolate a new field onto the evaluation points.

        Args:
            values: (N,) or (N, k) array of field values at scattered coordinates

        Returns:
            (M,) or (M, k) array of interpolated values. Points outside the convex hull are set to NaN.

        """
        values = np.asarray(values)
        out_shape = (self.n_eval,) if values.ndim == 1 else (self.n_eval, values.shape[1])
        out = np.full(out_shape, np.nan, dtype=values.dtype)

        if values.ndim == 1:
            out[self.valid] = (self.bary[self.valid] * values[self.vidx[self.valid]]).sum(axis=1)
        else:
            out[self.valid] = (self.bary[self.valid, :, None] * values[self.vidx[self.valid]]).sum(axis=1)

        return out


if __name__ == "__main__":
    pass
