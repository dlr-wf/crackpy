"""Integration Contours define resolved contour coordinates, traversal order,
and segment geometry for line-integral evaluation."""

from dataclasses import dataclass

import numpy as np


def _owned_read_only_array(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class IntegrationContour:
    """Store one resolved Integration Contour in established traversal order.

    Attributes:
        origin: Crack-tip-coordinate origin ``(x, y)`` in mm.
        nodes: Ordered contour-node coordinates with shape ``(n_nodes, 2)``.
        integration_points: Segment midpoint and increment rows
            ``[midpoint_x, midpoint_y, delta_x, delta_y]``.
        number_of_nodes: Resolved nominal node count.
        tick_size: Resolved contour tick size in mm.
    """

    origin: tuple[float, float]
    nodes: np.ndarray
    integration_points: np.ndarray
    number_of_nodes: int
    tick_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _owned_read_only_array(self.nodes))
        object.__setattr__(
            self,
            "integration_points",
            _owned_read_only_array(self.integration_points),
        )


@dataclass(frozen=True)
class ContourSet:
    """Store an immutable ordered collection of completed Integration Contours.

    Attributes:
        contours: Completed contours in evaluation order.
    """

    contours: tuple[IntegrationContour, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contours", tuple(self.contours))


@dataclass(frozen=True, eq=False)
class IntegrationContourGeometry:
    """Store immutable prepared geometry for one Integration Contour.

    Attributes:
        evaluation_points: Contour-segment midpoint coordinates.
        positive_x_shifted_evaluation_points: Midpoints shifted in positive x.
        negative_x_shifted_evaluation_points: Midpoints shifted in negative x.
        combined_evaluation_points: Base, positive-shifted, and
            negative-shifted midpoint coordinates in sampling order.
        relative_evaluation_points: Base midpoints relative to the contour origin.
        relative_positive_x_shifted_evaluation_points: Positive-shifted
            midpoints relative to the contour origin.
        relative_negative_x_shifted_evaluation_points: Negative-shifted
            midpoints relative to the contour origin.
        polar_radii: Polar radii of relative base midpoints in mm.
        polar_angles: Polar angles of relative base midpoints in radians.
        x_shift: Exact horizontal sampling shift in mm.
        segment_lengths: Positive contour-segment lengths ``ds`` in mm.
        segment_dy: Signed vertical increments in contour order in mm.
        outward_unit_normals: Outward unit normals in contour order.
        reference_point: Established Stress-Difference Method sampling point.
    """

    evaluation_points: np.ndarray
    positive_x_shifted_evaluation_points: np.ndarray
    negative_x_shifted_evaluation_points: np.ndarray
    combined_evaluation_points: np.ndarray
    relative_evaluation_points: np.ndarray
    relative_positive_x_shifted_evaluation_points: np.ndarray
    relative_negative_x_shifted_evaluation_points: np.ndarray
    polar_radii: np.ndarray
    polar_angles: np.ndarray
    x_shift: float
    segment_lengths: np.ndarray
    segment_dy: np.ndarray
    outward_unit_normals: np.ndarray
    reference_point: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_shift", float(self.x_shift))
        for field_name in self.__dataclass_fields__:
            if field_name == "x_shift":
                continue
            object.__setattr__(
                self,
                field_name,
                _owned_read_only_array(getattr(self, field_name)),
            )


def build_rectangular_integration_contour(
    *,
    origin_x: float,
    origin_y: float,
    size_left: float,
    size_right: float,
    size_bottom: float,
    size_top: float,
    tick_size: float | None,
    number_of_nodes: int | None,
    top_offset: float,
    bottom_offset: float,
) -> IntegrationContour:
    """Build a resolved rectangular Integration Contour in CrackPy traversal order.

    Args:
        origin_x: Contour origin x-coordinate in mm.
        origin_y: Contour origin y-coordinate in mm.
        size_left: Left contour bound relative to the origin in mm.
        size_right: Right contour bound relative to the origin in mm.
        size_bottom: Bottom contour bound relative to the origin in mm.
        size_top: Top contour bound relative to the origin in mm.
        tick_size: Requested contour tick size in mm, or ``None``.
        number_of_nodes: Requested nominal node count, or ``None``.
        top_offset: Upper crack-face contour offset in mm.
        bottom_offset: Lower crack-face contour offset in mm.

    Returns:
        An immutable completed Integration Contour.

    Raises:
        ValueError: If neither discretization input is supplied or the lower
            left contour section receives zero nodes.
        ZeroDivisionError: If an explicit zero nominal node count is supplied.
    """
    length_bottom_left = bottom_offset - size_bottom
    length_bottom = size_right - size_left
    length_right = size_top - size_bottom
    length_top = length_bottom
    length_top_left = size_top - top_offset
    total_length = (
        length_bottom_left
        + length_bottom
        + length_right
        + length_top
        + length_top_left
    )

    if number_of_nodes is None:
        if tick_size is None:
            raise ValueError(
                "Either number of edges or integral tick size needs to be specified!"
            )
        resolved_number_of_nodes = int(total_length / tick_size) + 1
        resolved_tick_size = tick_size
    else:
        resolved_number_of_nodes = number_of_nodes
        resolved_tick_size = total_length / number_of_nodes

    nodes_x = []
    nodes_y = []

    # The open contour starts on the lower crack face and follows the lower-left,
    # bottom, right, top, and upper-left sections to the upper crack face.
    section_nodes = int(
        length_bottom_left / total_length * resolved_number_of_nodes
    )
    if section_nodes == 0:
        raise ValueError(
            "Number of nodes is zero. Choose a smaller integral tick size!"
        )
    for index in range(section_nodes + 1):
        nodes_x.append(origin_x + size_left)
        nodes_y.append(
            origin_y
            + bottom_offset
            - length_bottom_left / section_nodes * index
        )

    section_nodes = int(length_bottom / total_length * resolved_number_of_nodes)
    for index in range(1, section_nodes + 1):
        nodes_x.append(
            origin_x + size_left + length_bottom / section_nodes * index
        )
        nodes_y.append(origin_y + size_bottom)

    section_nodes = int(length_right / total_length * resolved_number_of_nodes)
    for index in range(1, section_nodes + 1):
        nodes_x.append(origin_x + size_right)
        nodes_y.append(
            origin_y + size_bottom + length_right / section_nodes * index
        )

    section_nodes = int(length_top / total_length * resolved_number_of_nodes)
    for index in range(1, section_nodes + 1):
        nodes_x.append(
            origin_x + size_right - length_top / section_nodes * index
        )
        nodes_y.append(origin_y + size_top)

    section_nodes = int(length_top_left / total_length * resolved_number_of_nodes)
    for index in range(1, section_nodes + 1):
        nodes_x.append(origin_x + size_left)
        nodes_y.append(
            origin_y + size_top - length_top_left / section_nodes * index
        )

    nodes = np.column_stack((nodes_x, nodes_y))
    increments = nodes[1:] - nodes[:-1]
    midpoints = nodes[:-1] + increments / 2.0
    integration_points = np.column_stack((midpoints, increments))
    return IntegrationContour(
        origin=(origin_x, origin_y),
        nodes=nodes,
        integration_points=integration_points,
        number_of_nodes=resolved_number_of_nodes,
        tick_size=resolved_tick_size,
    )


def prepare_integration_contour_geometry(
    contour: IntegrationContour,
    *,
    x_shift: float = 0.0,
) -> IntegrationContourGeometry:
    """Prepare immutable segment and shifted sampling geometry for a contour.

    Args:
        contour: Completed Integration Contour to prepare.
        x_shift: Horizontal sampling shift in mm.

    Returns:
        Prepared contour geometry in the established sampling order.
    """
    evaluation_points = contour.integration_points[:, :2].copy()
    positive_x_shifted = evaluation_points + np.asarray([x_shift, 0.0])
    negative_x_shifted = evaluation_points - np.asarray([x_shift, 0.0])
    combined = np.r_[evaluation_points, positive_x_shifted, negative_x_shifted]
    origin = np.asarray([contour.origin])
    relative = evaluation_points - origin
    relative_positive = positive_x_shifted - origin
    relative_negative = negative_x_shifted - origin
    polar_radii = np.sqrt(relative[:, 0] ** 2.0 + relative[:, 1] ** 2.0)
    polar_angles = np.arctan2(relative[:, 1], relative[:, 0])
    segment_dy = contour.integration_points[:, 3].copy()
    segment_lengths = np.linalg.norm(contour.integration_points[:, 2:4], axis=1)
    # Rotating each oriented [dx, dy] segment clockwise gives the outward
    # direction [dy, -dx], which is normalized by the positive arc length ds.
    normal_vectors = np.c_[
        contour.integration_points[:, 3],
        -contour.integration_points[:, 2],
    ].astype(
        np.result_type(contour.integration_points.dtype, np.float64),
        copy=False,
    )
    normal_lengths = np.linalg.norm(normal_vectors, axis=1)
    outward_unit_normals = normal_vectors / normal_lengths[:, np.newaxis]
    reference_point = np.asarray(
        [[np.max(contour.integration_points[:, 0]), 0.0]]
    )
    return IntegrationContourGeometry(
        evaluation_points=evaluation_points,
        positive_x_shifted_evaluation_points=positive_x_shifted,
        negative_x_shifted_evaluation_points=negative_x_shifted,
        combined_evaluation_points=combined,
        relative_evaluation_points=relative,
        relative_positive_x_shifted_evaluation_points=relative_positive,
        relative_negative_x_shifted_evaluation_points=relative_negative,
        polar_radii=polar_radii,
        polar_angles=polar_angles,
        x_shift=x_shift,
        segment_lengths=segment_lengths,
        segment_dy=segment_dy,
        outward_unit_normals=outward_unit_normals,
        reference_point=reference_point,
    )
