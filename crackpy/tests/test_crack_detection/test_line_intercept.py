"""Check line-intercept geometry against known paths and their reflections."""

import numpy as np
import pytest

from crackpy.crack_detection.line_intercept import CrackDetectionLineIntercept
from crackpy.input.input_data import InputData


def _detection(side='right', tip=6, window_size=3, slope=0.2, radius=50):
    x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(-4, 4, 41))
    data = InputData()
    data.coor_x = (x if side == 'right' else -x).ravel()
    data.coor_y = y.ravel()
    data.disp_x = np.zeros(x.size)
    data.disp_y = (np.tanh(y - slope * x) + 0.01 * y).ravel()
    # An isolated peak ahead of the tip must not satisfy a three-point plateau.
    data.eps_vm = np.where((x <= tip) | ((x == 9) & (window_size > 1)), 0.02, 0.0).ravel()
    return CrackDetectionLineIntercept(
        data, x_min=float(data.coor_x.min()), x_max=float(data.coor_x.max()),
        y_min=-4, y_max=4, tick_size_x=10 / 11, tick_size_y=8 / 41,
        window_size=window_size, side=side, angle_estimation_mm_radius=radius,
    )


@pytest.mark.parametrize('slope', [-0.2, 0.0, 0.2])
def test_mirrored_plateau_tip_path_and_angle(slope):
    right = _detection(slope=slope)
    left = _detection(side='left', slope=slope)
    right.run()
    left.run()

    np.testing.assert_allclose(right.crack_tip, [6, 6 * slope], atol=1e-6)
    np.testing.assert_allclose(left.crack_tip, [-6, 6 * slope], atol=1e-6)
    expected_path = np.column_stack((np.arange(6), slope * np.arange(6)))
    np.testing.assert_allclose(right.crack_path, expected_path, atol=1e-6)
    np.testing.assert_allclose(left.crack_path, expected_path * [-1, 1], atol=1e-6)
    angle = np.degrees(np.arctan(slope))
    assert right.crack_angle == pytest.approx(angle, abs=1e-5)
    assert left.crack_angle == pytest.approx(180 - angle, abs=1e-5)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_plateau_reaching_window_boundary_retains_tip(side):
    detection = _detection(side=side, tip=10)
    detection.run()
    direction = 1 if side == 'right' else -1
    np.testing.assert_allclose(detection.crack_tip, [direction * 10, 2], atol=1e-6)
    assert detection.crack_path.shape == (10, 2)
    assert detection.crack_angle == pytest.approx(
        np.degrees(np.arctan(0.2)) if side == 'right' else 180 - np.degrees(np.arctan(0.2)),
        abs=1e-5,
    )


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('tip', [0, 1, 2])
def test_tip_survives_insufficient_angle_support(side, tip):
    detection = _detection(side=side, tip=tip, window_size=1)
    detection.run()
    direction = 1 if side == 'right' else -1
    np.testing.assert_allclose(detection.crack_tip, [direction * tip, 0.2 * tip], atol=1e-6)
    assert detection.crack_path.shape == (tip, 2)
    assert np.isnan(detection.crack_angle)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_no_plateau_returns_unavailable_geometry(side):
    detection = _detection(side=side, tip=-1)
    detection.run()
    assert np.isnan(detection.crack_tip).all()
    assert np.isnan(detection.crack_path).all()
    assert np.isnan(detection.crack_angle)


def test_line_intercept_rejects_unsupported_side():
    with pytest.raises(ValueError, match='side'):
        _detection(side='centre')


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('radius', [0.1, 1, 2])
def test_small_angle_window_retains_tip(side, radius):
    detection = _detection(side=side, radius=radius)
    detection.run()
    assert np.isfinite(detection.crack_tip).all()
    assert np.isnan(detection.crack_angle)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_rerun_clears_geometry_when_plateau_disappears(side):
    detection = _detection(side=side)
    detection.run()
    detection.data.eps_vm[:] = 0
    detection.run()
    assert np.isnan(detection.crack_tip).all()
    assert np.isnan(detection.crack_path).all()
    assert np.isnan(detection.crack_angle)
