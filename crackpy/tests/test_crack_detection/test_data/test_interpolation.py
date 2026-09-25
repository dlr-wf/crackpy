from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import griddata

import crackpy.crack_detection.data.datapreparation as dp
import crackpy.crack_detection.data.interpolation as interpolation_module
from crackpy.crack_detection.data.interpolation import interpolate, interpolate_on_array
from crackpy.crack_detection.utils.utilityfunctions import get_nodemaps_and_stage_nums


class TestInterpolation(unittest.TestCase):

    def test_interpolate_uses_one_scattered_interpolation_for_all_fields(self):
        coor_x = np.array([0.0, 2.0, 0.0, 2.0, 1.0])
        coor_y = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])
        frame = SimpleNamespace(
            coor_x=coor_x,
            coor_y=coor_y,
            disp_x=2.0 * coor_x + 3.0 * coor_y + 1.0,
            disp_y=-1.5 * coor_x + 0.5 * coor_y - 2.0,
            eps_vm=0.25 * coor_x - 0.75 * coor_y + 0.5,
        )

        with patch.object(interpolation_module, 'griddata', wraps=griddata) as interpolation_spy:
            _, displacements, eps_vm = interpolate(frame, size=2.0, pixels=2)

        self.assertEqual(interpolation_spy.call_count, 1)
        np.testing.assert_allclose(
            displacements,
            np.array([
                [[-2.0, 2.0], [4.0, 8.0]],
                [[-2.5, -5.5], [-1.5, -4.5]],
            ]),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            eps_vm,
            np.array([[1.25, 1.75], [-0.25, 0.25]]),
            rtol=0.0,
            atol=1e-12,
        )

    def test_interpolate_preserves_two_sided_field_specific_nan_parity(self):
        base_x = np.array([0.0, 2.0, 0.0, 2.0, 1.0])
        coor_y = np.array([-1.0, -1.0, 1.0, 1.0, 0.0])

        for size in (2.0, -2.0):
            with self.subTest(size=size):
                coor_x = base_x if size > 0 else -base_x
                frame = SimpleNamespace(
                    coor_x=coor_x,
                    coor_y=coor_y,
                    disp_x=np.array([np.nan, 2.0, 4.0, 8.0, 1.0]),
                    disp_y=np.array([-2.5, np.nan, -1.5, -4.5, -3.0]),
                    eps_vm=np.array([1.25, 1.75, np.nan, 0.25, 0.5]),
                )
                x_axis = np.linspace(min(size, 0.0), max(size, 0.0), 5)
                y_axis = np.linspace(-1.0, 1.0, 5)
                x_grid, y_grid = np.meshgrid(x_axis, y_axis)
                expected_displacements = np.array([
                    griddata((coor_x, coor_y), frame.disp_x, (x_grid, y_grid)),
                    griddata((coor_x, coor_y), frame.disp_y, (x_grid, y_grid)),
                ])
                expected_eps_vm = griddata(
                    (coor_x, coor_y),
                    frame.eps_vm,
                    (x_grid, y_grid),
                )
                if size < 0:
                    expected_displacements = np.flip(expected_displacements, axis=2)
                    expected_displacements[0] *= -1.0
                    expected_eps_vm = np.fliplr(expected_eps_vm)

                _, displacements, eps_vm = interpolate(frame, size=size, pixels=5)

                for actual, expected in (
                    (displacements, expected_displacements),
                    (eps_vm, expected_eps_vm),
                ):
                    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-12,
                        atol=1e-12,
                        equal_nan=True,
                    )

    def test_interpolation(self):
        root = Path(__file__).resolve().parents[4]
        origin = root / 'test_data' / 'crack_detection' / 'raw'
        side = 'left'
        size = 70

        stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(str(origin / 'Nodemaps'), ['7'])

        # import
        inputs, _ = dp.import_data(nodemaps=stages_to_nodemaps,
                                   data_path=str(origin),
                                   side=side,
                                   exists_target=False)
        # interpolate
        interp_size = size if side == 'right' else size * -1
        interp_coors, interp_disps, interp_eps_vm = interpolate_on_array(input_by_nodemap=inputs,
                                                                         interp_size=interp_size,
                                                                         pixels=256)
        # tests
        self.assertIsInstance(interp_coors, dict)
        self.assertIsInstance(interp_disps, dict)
        self.assertIsInstance(interp_eps_vm, dict)

        for coors in interp_coors.values():
            self.assertIsInstance(coors, np.ndarray)
            self.assertEqual(coors.shape, (2, 256, 256))
            self.assertAlmostEqual(coors[0, 0, 1], -0.2745098, delta=1e-6)

        for disps in interp_disps.values():
            self.assertIsInstance(disps, np.ndarray)
            self.assertEqual(disps.shape, (2, 256, 256))
            self.assertAlmostEqual(disps[0, 0, 0], -0.017540, delta=1e-6)

        for eps_vm in interp_eps_vm.values():
            self.assertIsInstance(eps_vm, np.ndarray)
            self.assertEqual(eps_vm.shape, (256, 256))
            self.assertAlmostEqual(eps_vm[0, 0], 0.001070, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
