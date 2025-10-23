"""Module for crack tip information wrapper."""

class CrackTipInfo:
    """Wrapper for crack tip information.

    Methods:
        * set_manually - manually redefine crack tip information

    """

    def __init__(
            self,
            crack_tip_x: float = None,
            crack_tip_y: float = None,
            crack_tip_angle: float = None,
            left_or_right: str = None
    ):
        """Initialize crack tip info with provided attributes.

        Args:
            crack_tip_x: x-coordinate of the actual crack tip
            crack_tip_y: y-coordinate of the actual crack tip
            crack_tip_angle: angle of crack path between 0 and 180 degree
            left_or_right: either 'l' or 'r'

        """
        self.crack_tip_x = crack_tip_x
        self.crack_tip_y = crack_tip_y
        self.crack_tip_angle = crack_tip_angle
        self.left_or_right = left_or_right

    def set_manually(self, crack_tip_x: float, crack_tip_y: float, crack_tip_angle: float, left_or_right: str):
        """Alternatively coordinates may be given externally, e.g. from the crack detection module.

        Args:
            crack_tip_x: x-coordinate of the actual crack tip
            crack_tip_y: y-coordinate of the actual crack tip
            crack_tip_angle: angle of crack path between 0 and 180 degree
            left_or_right: either 'l' or 'r'

        """
        self.crack_tip_x = crack_tip_x
        self.crack_tip_y = crack_tip_y
        self.crack_tip_angle = crack_tip_angle
        self.left_or_right = left_or_right
