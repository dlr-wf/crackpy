from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FileStructure:
    """Base class for file structure definitions."""

    def __init__(self) -> None:
        pass


class NodemapStructure(FileStructure):
    """Structure definition for nodemap files.

    Defines the row length and column indices for various data fields in nodemap files.

    Attributes:
        row_length: Number of columns in each row
        index_facet_id: Column index for facet ID
        index_coor_x: Column index for x-coordinate
        index_coor_y: Column index for y-coordinate
        index_coor_z: Column index for z-coordinate
        index_disp_x: Column index for x-displacement
        index_disp_y: Column index for y-displacement
        index_disp_z: Column index for z-displacement
        index_eps_x: Column index for x-strain
        index_eps_y: Column index for y-strain
        index_eps_xy: Column index for xy-strain
        strain_is_percent: Whether strain values are in percent
        is_fem: Whether data is from FEM simulation

    """

    def __init__(
            self,
            row_length: int = 10,
            index_facet_id: int = 0,
            index_coor_x: int = 1,
            index_coor_y: int = 2,
            index_coor_z: int = 3,
            index_disp_x: int = 4,
            index_disp_y: int = 5,
            index_disp_z: int = 6,
            index_eps_x: int = 7,
            index_eps_y: int = 8,
            index_eps_xy: int = 9,
            strain_is_percent: bool = True,
            is_fem: bool = False
    ) -> None:
        super().__init__()
        self.row_length = row_length
        self.index_facet_id = index_facet_id
        self.index_coor_x = index_coor_x
        self.index_coor_y = index_coor_y
        self.index_coor_z = index_coor_z
        self.index_disp_x = index_disp_x
        self.index_disp_y = index_disp_y
        self.index_disp_z = index_disp_z
        self.index_eps_x = index_eps_x
        self.index_eps_y = index_eps_y
        self.index_eps_xy = index_eps_xy
        self.strain_is_percent = strain_is_percent
        self.is_fem = is_fem


class DataFile:
    """Base class for data files with defined structure.

    Attributes:
        name: Data file name
        folder: Path to data file
        structure: File structure definition
        project_name: Project name
        specimen_name: Specimen name

    """

    def __init__(self, name: str, folder: str | Path = None,
                 structure: FileStructure = None,
                 project_name: str = "example project",
                 specimen_name: str = "specimen name") -> None:
        """Data File class for data files which are usually .txt-files with a defined file structure given as the
       'structure' argument.

        Args:
            name: data file name as string
            folder: path to data file
            structure: optional given structure for the data file
            project_name: optional given project to which the data refers
            specimen_name: optional given specimen name to which the data refers

        """

        self.name = name
        self.folder = Path(folder) if folder is not None else None
        self.project_name = project_name
        self.specimen_name = specimen_name
        self.structure = structure

        # use lazy logging to avoid eager string formatting
        logger.debug("DataFile initialized: %s in %s", name, folder)


class Nodemap(DataFile):
    """Nodemap File class for DIC or FEM nodemap data.

    Attributes:
        name: Nodemap file name
        folder: Path to nodemap file
        structure: Nodemap structure definition
        project_name: Project name
        specimen_name: Specimen name

    """

    def __init__(self, name: str, folder: str | Path,
                 structure: NodemapStructure = NodemapStructure(),
                 project_name: str = "example project",
                 specimen_name: str = "specimen name") -> None:
        """Nodemap File class.

        Args:
            name: data file name as string
            folder: path to data file
            structure: optional given structure for the data file
            project_name: optional given project name
            specimen_name: optional given specimen name

        """
        super().__init__(name, folder, structure, project_name, specimen_name)
