"""Verify Integration Contours and their mutable facade projections."""

import re

import numpy as np
import pytest

from crackpy.fracture_analysis.line_integrals import ContourSet, IntegrationContour
from crackpy.fracture_analysis.line_integrals.contours import (
    build_rectangular_integration_contour,
    prepare_integration_contour_geometry,
)
from crackpy.fracture_analysis.line_integration import IntegrationPath, PathProperties


def _rectangular_contour(
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    tick_size: float | None = None,
    number_of_nodes: int | None = 8,
) -> IntegrationContour:
    return build_rectangular_integration_contour(
        origin_x=origin_x,
        origin_y=origin_y,
        size_left=-1.0,
        size_right=1.0,
        size_bottom=-1.0,
        size_top=1.0,
        tick_size=tick_size,
        number_of_nodes=number_of_nodes,
        top_offset=0.0,
        bottom_offset=0.0,
    )


def test_rectangular_contour_preserves_node_and_integration_point_order():
    contour = _rectangular_contour()
    expected_nodes = np.asarray(
        [
            [-1.0, 0.0],
            [-1.0, -1.0],
            [0.0, -1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [-1.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    expected_integration_points = np.c_[
        (expected_nodes[:-1] + expected_nodes[1:]) / 2.0,
        expected_nodes[1:] - expected_nodes[:-1],
    ]

    np.testing.assert_array_equal(contour.nodes, expected_nodes)
    np.testing.assert_array_equal(
        contour.integration_points,
        expected_integration_points,
    )


def test_contour_geometry_preserves_signed_dy_positive_ds_and_outward_normals():
    contour = _rectangular_contour()
    geometry = prepare_integration_contour_geometry(contour)

    np.testing.assert_array_equal(
        geometry.segment_dy,
        contour.integration_points[:, 3],
    )
    np.testing.assert_allclose(geometry.segment_lengths, np.ones(8))
    np.testing.assert_array_equal(
        geometry.outward_unit_normals,
        np.asarray(
            [
                [-1.0, 0.0],
                [0.0, -1.0],
                [0.0, -1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_contour_normals_preserve_promoted_normalization_precision(dtype):
    integration_points = np.asarray(
        [
            [0.0, 0.0, 0.1, 0.3],
            [0.1, 0.3, -0.7, 0.2],
        ],
        dtype=dtype,
    )
    contour = IntegrationContour(
        origin=(0.0, 0.0),
        nodes=np.asarray([[0.0, 0.0], [0.1, 0.3], [-0.6, 0.5]], dtype=dtype),
        integration_points=integration_points,
        number_of_nodes=2,
        tick_size=1.0,
    )
    geometry = prepare_integration_contour_geometry(contour)
    promoted_directions = np.c_[
        integration_points[:, 2:4],
        np.zeros(len(integration_points)),
    ]
    previous_normal_vectors = np.cross(
        promoted_directions,
        [0.0, 0.0, 1.0],
    )
    expected_normals = (previous_normal_vectors / np.linalg.norm(previous_normal_vectors, axis=1)[:, np.newaxis])[:, :2]

    np.testing.assert_array_equal(geometry.outward_unit_normals, expected_normals)


def test_number_of_nodes_takes_precedence_and_facade_projection_is_mutable():
    properties = PathProperties(-1.0, 1.0, -1.0, 1.0, 99.0, 8, 0.0, 0.0)
    path = IntegrationPath(path_properties=properties)

    assert properties.number_of_nodes == 8
    assert properties.tick_size == 1.0
    path.nodes[0][0] = 123.0
    path.int_points[0, 0] = 456.0
    assert path.nodes[0][0] == 123.0
    assert path.int_points[0, 0] == 456.0


def test_tick_size_derives_nominal_node_count_and_preserves_tick_size():
    properties = PathProperties(-1.0, 1.0, -1.0, 1.0, 1.0, None, 0.0, 0.0)
    IntegrationPath(path_properties=properties)

    assert properties.number_of_nodes == 9
    assert properties.tick_size == 1.0


def test_nominal_node_count_can_differ_from_constructed_node_count():
    contour = _rectangular_contour(number_of_nodes=10)

    assert contour.number_of_nodes == 10
    assert contour.nodes.shape == (9, 2)
    assert contour.integration_points.shape == (8, 4)


@pytest.mark.parametrize(
    ("tick_size", "number_of_nodes", "error_type", "message"),
    [
        (
            None,
            None,
            ValueError,
            "Either number of edges or integral tick size needs to be specified!",
        ),
        (
            100.0,
            None,
            ValueError,
            "Number of nodes is zero. Choose a smaller integral tick size!",
        ),
        (None, 0, ZeroDivisionError, None),
    ],
)
def test_rectangular_contour_preserves_exact_construction_failures(
    tick_size,
    number_of_nodes,
    error_type,
    message,
):
    expected_message = None if message is None else f"^{re.escape(message)}$"
    with pytest.raises(error_type, match=expected_message):
        _rectangular_contour(
            tick_size=tick_size,
            number_of_nodes=number_of_nodes,
        )


def test_translated_contour_reference_point_keeps_global_zero_y_coordinate():
    contour = _rectangular_contour(origin_x=5.0, origin_y=7.0)
    geometry = prepare_integration_contour_geometry(contour)

    np.testing.assert_array_equal(geometry.reference_point, [[6.0, 0.0]])


def test_contour_arrays_are_read_only_and_do_not_alias_facade_projections():
    properties = PathProperties(-1.0, 1.0, -1.0, 1.0, None, 8, 0.0, 0.0)
    path = IntegrationPath(path_properties=properties)
    contour = build_rectangular_integration_contour(
        origin_x=path.origin_x,
        origin_y=path.origin_y,
        size_left=properties.size_left,
        size_right=properties.size_right,
        size_bottom=properties.size_bottom,
        size_top=properties.size_top,
        tick_size=None,
        number_of_nodes=8,
        top_offset=properties.top_offset,
        bottom_offset=properties.bottom_offset,
    )
    geometry = prepare_integration_contour_geometry(contour)

    contour_arrays = [
        contour.nodes,
        contour.integration_points,
        geometry.segment_lengths,
        geometry.segment_dy,
        geometry.outward_unit_normals,
        geometry.reference_point,
    ]
    assert all(not array.flags.writeable for array in contour_arrays)

    path.nodes[0][0] = 123.0
    path.int_points[0, 0] = 456.0
    assert contour.nodes[0, 0] == -1.0
    assert contour.integration_points[0, 0] == -1.0


def test_contour_set_preserves_tuple_order():
    first = _rectangular_contour(origin_x=1.0)
    second = _rectangular_contour(origin_x=2.0)

    contours = ContourSet([first, second])

    assert contours.contours == (first, second)


def test_line_integrals_package_exports_only_completed_contour_values():
    from crackpy.fracture_analysis import line_integrals

    assert line_integrals.__all__ == [
        "IntegrationContour",
        "ContourSet",
        "IntegrationContourResultGeometry",
        "LineIntegralQuantities",
        "ContourWiseLineIntegralResult",
    ]
