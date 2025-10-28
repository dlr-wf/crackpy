"""Module for crack tip information wrapper."""
import logging

logger = logging.getLogger(__name__)



class CrackTipInfo:
    """Wrapper for crack tip information.

    Stores the crack tip position, angle, and side information.

    Attributes:
        crack_tip_x: x-coordinate of the crack tip in mm
        crack_tip_y: y-coordinate of the crack tip in mm
        crack_tip_angle: angle of crack path in degrees (0 to 180)
        left_or_right: side identifier ('l' or 'r')

    Methods:
        * set_manually - manually redefine crack tip information

    """

    def __init__(
            self,
            crack_tip_x: float = None,
            crack_tip_y: float = None,
            crack_tip_angle: float = None,
            left_or_right: str = None
    ) -> None:
        """Initialize crack tip info with provided attributes.

        Args:
            crack_tip_x: x-coordinate of the actual crack tip in mm
            crack_tip_y: y-coordinate of the actual crack tip in mm
            crack_tip_angle: angle of crack path between 0 and 180 degrees
            left_or_right: either 'l' or 'r'

        """
        self.crack_tip_x = crack_tip_x
        self.crack_tip_y = crack_tip_y
        self.crack_tip_angle = crack_tip_angle
        self.left_or_right = left_or_right

        logger.debug("CrackTipInfo initialized: x=%s, y=%s, angle=%s, side=%s", crack_tip_x, crack_tip_y, crack_tip_angle, left_or_right)

    def set_manually(self, crack_tip_x: float, crack_tip_y: float,
                     crack_tip_angle: float, left_or_right: str) -> None:
        """Alternatively coordinates may be given externally, e.g. from the crack detection module.

        Args:
            crack_tip_x: x-coordinate of the actual crack tip in mm
            crack_tip_y: y-coordinate of the actual crack tip in mm
            crack_tip_angle: angle of crack path between 0 and 180 degrees
            left_or_right: either 'l' or 'r'

        """
        self.crack_tip_x = crack_tip_x
        self.crack_tip_y = crack_tip_y
        self.crack_tip_angle = crack_tip_angle
        self.left_or_right = left_or_right

        logger.debug("CrackTipInfo updated manually: x=%s, y=%s, angle=%s, side=%s", crack_tip_x, crack_tip_y, crack_tip_angle, left_or_right)