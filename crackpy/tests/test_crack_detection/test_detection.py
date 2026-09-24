"""Check supported sides of neural-network detection windows."""

import pytest

from crackpy.crack_detection.detection import CrackDetection


@pytest.mark.parametrize('side,expected', [('right', 70), ('left', -70)])
def test_side_sets_signed_interpolation_width(side, expected):
    assert CrackDetection(side=side, device='cpu').interp_size == expected


def test_detection_rejects_unsupported_side():
    with pytest.raises(ValueError, match='side'):
        CrackDetection(side='centre', device='cpu')
