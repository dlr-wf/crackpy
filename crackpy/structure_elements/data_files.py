import os


class FileStructure:
    def __init__(self):
        pass


class NodemapStructure(FileStructure):
    def __init__(
            self,
            row_length=10,
            index_facet_id=0,
            index_coor_x=1,
            index_coor_y=2,
            index_coor_z=3,
            index_disp_x=4,
            index_disp_y=5,
            index_disp_z=6,
            index_eps_x=7,
            index_eps_y=8,
            index_eps_xy=9,
            strain_is_percent=True,
            is_fem=False
    ):
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
    def __init__(self, name: str, folder: str or os.path = None,
                 structure: FileStructure = None,
                 project_name: str = "example project",
                 specimen_name: str = "specimen name"):
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
        self.folder = folder
        self.project_name = project_name
        self.specimen_name = specimen_name
        self.structure = structure


class Nodemap(DataFile):
    def __init__(self, name: str, folder: str or os.path,
                 structure: NodemapStructure = NodemapStructure(),
                 project_name: str = "example project",
                 specimen_name: str = "specimen name"):
        """Nodemap File class.

        Args:
            name: data file name as string
            folder: path to data file
            structure: optional given structure for the data file
            project_name: optional given project name
            specimen_name: optional given specimen name

        """
        super().__init__(name, folder, structure, project_name, specimen_name)
