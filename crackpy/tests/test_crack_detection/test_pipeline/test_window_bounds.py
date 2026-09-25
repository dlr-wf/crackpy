"""Exercise moving detection windows through nodemap loading and interpolation."""

import numpy as np
import pytest
import torch

from crackpy.crack_detection.detection import CrackDetection
from crackpy.crack_detection.pipeline import pipeline as pipeline_module


class _TipSequence(torch.nn.Module):
    """Supply deterministic tip segmentations to the real detection pipeline."""

    def __init__(self, pixels):
        super().__init__()
        self.pixels = iter(pixels)

    def forward(self, inputs):
        """Return the next segmentation and the unused regression output."""
        mask = inputs.new_zeros((1, 1, 256, 256))
        row, column = next(self.pixels)
        mask[:, :, row:row + 2, column:column + 2] = 1
        return mask, inputs.new_zeros((1, 2))


@pytest.fixture
def nodemaps(tmp_path):
    """Write three field snapshots covering both mirrored detection regions."""
    folder = tmp_path / 'nodemaps'
    folder.mkdir()
    x_grid, y_grid = np.meshgrid(np.linspace(-100, 100, 41), np.linspace(-50, 50, 21))
    x, y = x_grid.ravel(), y_grid.ravel()
    fields = np.column_stack((
        np.arange(x.size), x, y, np.zeros(x.size),
        x * 0.001, y * -0.0002, np.zeros(x.size),
        np.full(x.size, 0.1), np.full(x.size, -0.02), np.zeros(x.size),
    ))
    for stage in (1, 2, 3):
        np.savetxt(folder / f'field_{stage}.txt', fields, delimiter=';')
    return folder


@pytest.mark.parametrize(
    'side,boundary,start,pixels,expected_offsets',
    [
        pytest.param('right', (0, 70, -25, 25), (25, 0), [(126, 223)] * 3,
                     [(25, 0), (30, 0), (30, 0)], id='right-x-limit'),
        pytest.param('left', (0, 70, -25, 25), (25, 0), [(126, 223)] * 3,
                     [(-25, 0), (-30, 0), (-30, 0)], id='left-mirrored-x-limit'),
        pytest.param('right', (0, 70, -25, 25), (0, 0), [(223, 126)] * 3,
                     [(0, 0), (0, 5), (0, 5)], id='upper-y-limit'),
        pytest.param('left', (0, 70, -25, 25), (0, 0), [(31, 126)] * 3,
                     [(0, 0), (0, -5), (0, -5)], id='lower-y-limit'),
        pytest.param('right', (0, 70, -25, 25), (0, 0),
                     [(223, 126), (31, 126), (126, 126)],
                     [(0, 0), (0, 5), (0, -5)], id='y-can-return-from-boundary'),
        pytest.param('right', (0, 100, -50, 50), (0, 0), [(126, 255)] * 3,
                     [(0, 0), (20.15686274509804, 0), (40.31372549019608, 0)],
                     id='unrestricted-right-motion'),
        pytest.param('left', (0, 100, -50, 50), (0, 0), [(126, 255)] * 3,
                     [(0, 0), (-20.15686274509804, 0), (-40.31372549019608, 0)],
                     id='unrestricted-left-motion'),
        pytest.param('right', (0, 70, -25, 25), (25, 0), [(126, 126)] * 3,
                     [(25, 0), (25, 0), (25, 0)], id='tip-inside-tracking-deadband'),
        pytest.param('left', (10, 70, -25, 25), (10, 0), [(126, 255)] * 3,
                     [(-10, 0), (-30, 0), (-30, 0)], id='nonzero-inner-boundary'),
    ],
)
def test_detection_windows_stay_inside_boundary(
    nodemaps, tmp_path, monkeypatch, side, boundary, start, pixels, expected_offsets,
):
    """Following a tip keeps later sampling and the current plot in their correct windows."""
    sampled_offsets = []
    plotted_offsets = []
    interpolate = CrackDetection.interpolate

    def record_interpolation(detection, data):
        sampled_offsets.append(detection.offset)
        return interpolate(detection, data)

    def record_plot(**kwargs):
        plotted_offsets.append(kwargs['offset'])

    monkeypatch.setattr(CrackDetection, 'interpolate', record_interpolation)
    monkeypatch.setattr(pipeline_module, 'plot_prediction', record_plot)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    setup = pipeline_module.CrackDetectionSetup(
        specimen_size=200,
        sides=[side],
        detection_window_size=40,
        detection_boundary=boundary,
        start_offset=start,
        tip_only=True,
    )
    pipeline = pipeline_module.CrackDetectionPipeline(
        data_path=str(nodemaps),
        output_path=str(tmp_path / 'results'),
        tip_detector_model=_TipSequence(pixels),
        setup=setup,
    )

    results = pipeline.run_detection()

    np.testing.assert_allclose(sampled_offsets, expected_offsets, atol=1e-12)
    np.testing.assert_allclose(plotted_offsets, sampled_offsets, atol=1e-12)
    assert set(results[side]) == {1, 2, 3}
    x_min, x_max, y_min, y_max = boundary
    for offset_x, offset_y in sampled_offsets:
        if side == 'left':
            offset_x = -offset_x
        assert x_min <= offset_x <= x_max - 40
        assert y_min + 20 <= offset_y <= y_max - 20
