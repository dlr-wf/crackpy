from crackpy.structure_elements.material import Material
from crackpy.structure_elements.project import Project


class Specimen:
    """Specimen super class. This is basically a placeholder class so far. But the class shall be used in future
    versions to calculate theoretical values for stress intensity factors, stresses, etc.

    Methods:
        * set_reference_project - Define the referenced project

    """
    def __init__(self, name: str, material: Material or None, type_, width: float, thickness: float):
        """Initialize specimen class. This class is under current development. Thus most methods are actually not
           doing anything, but are available as placeholders

        Args:
            name: specimen name
            material: the material of the specimen
            width: width of the specimen [mm]
            thickness: thickness of the specimen [mm]

        """
        self.name = name
        self.material = material
        self.type_ = type_
        self.width = width
        self.thickness = thickness
        self.project = None

    def set_reference_project(self, project: Project):
        """Defines the project that contains the specimen

        Args:
            project:

        """
        self.project = project


class MTSpecimen(Specimen):
    """MT Specimen class.

    Methods:
        * calculate_sifs -

    """

    def __init__(self, specimen_name: str, material: Material, width: float, thickness: float):
        """Initializes class arguments.

        Args:
            specimen_name: name of the specimen
            material: material of the specimen
            width: width of the specimen [mm]
            thickness: thickness of the specimen [mm]

        """
        super().__init__(specimen_name, material, "MT", width, thickness)
        self.name = specimen_name
        self.material = material


class CTSpecimen(Specimen):
    """CT Specimen class."""
    def __init__(self, specimen_name: str, material: Material, width: float, thickness: float):
        """Initializes class arguments.

        Args:
            specimen_name: name of the specimen
            material: material of the specimen
            width: width of the specimen [mm]
            thickness: thickness of the specimen [mm]

        """
        super().__init__(specimen_name, material, "CT", width, thickness)
        self.name = specimen_name
        self.material = material


class SETSpecimen(Specimen):
    def __init__(self, specimen_name: str, material: Material, width: float, thickness: float):
        """Initializes class arguments.

        Args:
            specimen_name: name of the specimen
            material: material of the specimen
            width: width of the specimen [mm]
            thickness: thickness of the specimen [mm]

        """
        super().__init__(specimen_name, material, "SET", width, thickness)
        self.name = specimen_name
        self.material = material


class CruciformSpecimen(Specimen):
    def __init__(self, specimen_name: str, material: Material, width_x: float, width_y: float, thickness: float):
        """Initializes class arguments.

        Args:
            specimen_name: name of the specimen
            material: material of the specimen
            width_x: width of the specimen in x direction [mm]
            width_y: width of the specimen in y direction [mm]
            thickness: thickness of the specimen [mm]

        """
        super().__init__(specimen_name, material, "Cruciform", width_x, thickness)
        self.name = specimen_name
        self.material = material
        self.width_x = width_x
        self.width_y = width_y
