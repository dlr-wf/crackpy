import unittest

import numpy as np
from scipy.interpolate import griddata

from crackpy.fracture_analysis.utils import ReusableLinearInterpolator


class TestPrecomputedInterpolatorVsGriddata(unittest.TestCase):

    @classmethod
    def setUp(cls):
        rng = np.random.default_rng(42)
        N = 300

        # scattered points in [-1, 1]^2
        cls.x = rng.uniform(-1.0, 1.0, size=N)
        cls.y = rng.uniform(-1.0, 1.0, size=N)

        # evaluation grid slightly larger to ensure some points are outside convex hull
        gx = np.linspace(-1.2, 1.2, 64)
        gy = np.linspace(-1.2, 1.2, 64)
        X, Y = np.meshgrid(gx, gy)
        cls.eval_pts = np.c_[X.ravel(), Y.ravel()]

        # scalar field and a vector-valued field (k=3) defined smoothly
        def f_scalar(x, y):
            return np.sin(np.pi * x) * np.cos(np.pi * y) + 0.1 * x

        cls.val_scalar = f_scalar(cls.x, cls.y)

        k1 = f_scalar(cls.x, cls.y)
        k2 = cls.x ** 2 - cls.y ** 2
        k3 = np.exp(-((cls.x * 1.5) ** 2 + (cls.y * 1.5) ** 2))
        cls.val_vector = np.c_[k1, k2, k3]

        # build interpolator once
        cls.ip = ReusableLinearInterpolator(cls.x, cls.y, cls.eval_pts)

    def test_scalar_linear_matches_griddata(self):
        gd = griddata((self.x, self.y), self.val_scalar, self.eval_pts, method='linear')
        ip = self.ip.interpolate(self.val_scalar)

        # inside hull: masks should match and values should agree tightly
        inside = self.ip.valid
        self.assertTrue(np.array_equal(inside, ~np.isnan(gd)))

        self.assertTrue(np.allclose(ip[inside], gd[inside], rtol=1e-12, atol=1e-12))

        # outside hull: both should be NaN
        outside = ~inside
        if np.any(outside):
            self.assertTrue(np.all(np.isnan(ip[outside])))
            self.assertTrue(np.all(np.isnan(gd[outside])))

    def test_vector_linear_matches_griddata(self):
        gd = griddata((self.x, self.y), self.val_vector, self.eval_pts, method='linear')
        ip = self.ip.interpolate(self.val_vector)

        # inside hull per component
        inside = self.ip.valid
        self.assertTrue(np.array_equal(inside[:, None], ~np.isnan(gd).any(axis=1, keepdims=True)))

        self.assertTrue(np.allclose(ip[inside], gd[inside], rtol=1e-12, atol=1e-12))

        # outside hull: all components NaN
        outside = ~inside
        if np.any(outside):
            self.assertTrue(np.all(np.isnan(ip[outside])))
            self.assertTrue(np.all(np.isnan(gd[outside])))

    def test_constant_field(self):
        rng = np.random.default_rng(1)
        N = 200
        x = rng.uniform(-1, 1, size=N)
        y = rng.uniform(-1, 1, size=N)
        gx = np.linspace(-1.1, 1.1, 40)
        gy = np.linspace(-1.1, 1.1, 40)
        X, Y = np.meshgrid(gx, gy)
        eval_pts = np.c_[X.ravel(), Y.ravel()]

        const_val = 3.14159
        vals = np.full(N, const_val)

        ip = ReusableLinearInterpolator(x, y, eval_pts)
        out = ip.interpolate(vals)
        gd = griddata((x, y), vals, eval_pts, method='linear')

        inside = ip.valid
        # inside hull should equal constant
        self.assertTrue(np.allclose(out[inside], const_val))
        self.assertTrue(np.allclose(gd[inside], const_val))
        # outside should be NaN for both
        outside = ~inside
        if np.any(outside):
            self.assertTrue(np.all(np.isnan(out[outside])))
            self.assertTrue(np.all(np.isnan(gd[outside])))

    def test_eval_at_sample_points_returns_original_values(self):
        rng = np.random.default_rng(2)
        N = 150
        x = rng.uniform(-0.9, 0.9, size=N)
        y = rng.uniform(-0.9, 0.9, size=N)
        vals = np.sin(np.pi * x)  # scalar field

        # evaluate exactly at the sample points
        eval_pts = np.c_[x, y]

        ip = ReusableLinearInterpolator(x, y, eval_pts)
        out = ip.interpolate(vals)

        # All sample points lie in the convex hull; values should match exactly
        self.assertTrue(np.allclose(out, vals, rtol=0.0, atol=1e-12))

    def test_duplicate_points_match_griddata(self):
        rng = np.random.default_rng(3)
        N_unique = 120
        x_u = rng.uniform(-1, 1, size=N_unique)
        y_u = rng.uniform(-1, 1, size=N_unique)
        vals_u = np.cos(x_u) + y_u * 0.2

        # duplicate first 10 points
        x = np.concatenate([x_u, x_u[:10]])
        y = np.concatenate([y_u, y_u[:10]])
        vals = np.concatenate([vals_u, vals_u[:10]])

        gx = np.linspace(-1.05, 1.05, 50)
        gy = np.linspace(-1.05, 1.05, 50)
        X, Y = np.meshgrid(gx, gy)
        eval_pts = np.c_[X.ravel(), Y.ravel()]

        ip = ReusableLinearInterpolator(x, y, eval_pts)
        out = ip.interpolate(vals)
        gd = griddata((x, y), vals, eval_pts, method='linear')

        inside = ip.valid
        self.assertTrue(np.array_equal(inside, ~np.isnan(gd)))
        self.assertTrue(np.allclose(out[inside], gd[inside], rtol=1e-12, atol=1e-12))

    def test_insufficient_points_raises(self):
        # fewer than 3 non-collinear points cannot define a 2D triangulation
        x = np.array([0.0, 0.5])
        y = np.array([0.0, 0.0])  # collinear
        eval_pts = np.array([[0.1, 0.0]])

        # depending on implementation this may raise a QhullError or another Exception
        with self.assertRaises(Exception):
            ReusableLinearInterpolator(x, y, eval_pts)


if __name__ == "__main__":
    unittest.main()
