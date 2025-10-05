import numpy as np
from scipy.spatial import Delaunay

class ReusableLinearInterpolator:
    """
    Precompute Delaunay triangulation + barycentric weights for a fixed set of
    evaluation points (grid or arbitrary points). Reuse for many fields.

    by ChatGPT5
    """

    def __init__(self, coor_x: np.ndarray, coor_y: np.ndarray, eval_points: np.ndarray) -> None:
        """
        Args:
            coor_x, coor_y : (N,) 1D arrays of measurement/scattered coords (length N)
            eval_points    : (M, 2) array of target points where you want interpolation
        """
        pts = np.c_[coor_x, coor_y]   # (N, 2)
        tri = Delaunay(pts)

        simp = tri.find_simplex(eval_points)        # simplex index per eval point
        T = tri.transform[simp, :2]                 # affine transforms
        R = eval_points - tri.transform[simp, 2]    # shifted eval points
        bary12 = np.einsum('mij,mj->mi', T, R)      # first 2 barycentric coords

        self.bary: np.ndarray = np.c_[bary12, 1 - bary12.sum(axis=1)]  # full barycentric weights
        self.vidx: np.ndarray = tri.simplices[simp]             # vertex indices
        self.valid: np.ndarray = simp >= 0                      # inside convex hull mask
        self.n_eval: int = eval_points.shape[0]

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """
        Interpolate a new field onto eval_points.

        Args:
            values : (N,) or (N, k) array of field values at scattered coords

        Returns:
            (M,) or (M, k) array of interpolated values
        """
        values = np.asarray(values)
        out_shape = (self.n_eval,) if values.ndim == 1 else (self.n_eval, values.shape[1])
        out = np.full(out_shape, np.nan, dtype=values.dtype)

        if values.ndim == 1:
            out[self.valid] = (self.bary[self.valid] * values[self.vidx[self.valid]]).sum(axis=1)
        else:
            out[self.valid] = (self.bary[self.valid, :, None] * values[self.vidx[self.valid]]).sum(axis=1)

        return out