import os

from crackpy.structure_elements.specimen import Specimen
from crackpy.structure_elements.project import Project


class Experiment:
    """Structure class for one experiment

    Methods:
        * add_to_project - You can refer an experiment to a project.
        * load_from_xml - loads experiment data

    """

    def __init__(self, name: str, type_: str="fcp", specimen: Specimen or None=None):
        """Initialization of class arguments.

        Args:
            name: Name of the experiment.
            type_: The experiment type.
                   Currently only "fcp" is supported and will be used if type_ argument is left out.
            specimen:

        """
        if specimen is not None:
            self.specimen_no = specimen.name

        self.project = None

        self.possible_types = ["fcp"]

        # experiment params
        self.name = name
        self.type = self._set_type(type_)
        self.machine = None
        self.orientation = None
        self.project = None
        self.technician = None
        self.scientist = None
        self.initial_cracklength = None
        self.initial_potential = None
        self.max_force = None
        self.min_force = None
        self.frequency = None

        self.material = None
        self.distance_potential_sites = None
        self.specimen_height = None
        self.specimen_thickness = None
        self.specimen_width = None

    def add_to_project(self, project: Project):
        """ You can refer an experiment to a project.

        Args:
            project: The project to refer to

        """
        self.project = project

    def _set_type(self, type_):
        if type_ in self.possible_types:
            return type_
        else:
            raise NotImplementedError("The specified experiment type is not supported.\n"
                                      "Choose one of {self.possible_types}.")

    def load_from_xml(self, data_path: str or os.path):
        """Loads experiment data.

        Args:
            data_path: path to the xml file

        """
        if not os.path.exists(data_path):
            raise ValueError('Path or file for xml import not found')

        with open(data_path, mode='r') as file:
            lines_file = file.readlines()

            for index, line in enumerate(lines_file):
                if '<Name>' in line:
                    var_name = line.strip().strip("<Name>").strip("</").lower()
                    if var_name in self.__dict__:
                        var_value = lines_file[index + 2].strip().strip('<Value>').strip('</Value>')
                        if var_value.replace('.', '').isdigit():
                            if 'force' in var_name:
                                setattr(self, var_name, 1000 * float(var_value))
                            else:
                                setattr(self, var_name, float(var_value))
                        else:
                            setattr(self, var_name, var_value)


class Method:
    """We define a 'Method' as one specific method within one experiment, i.e. experiment may be fatigue crack growth
       and method can be e.g. 'DIC'.
    """
    def __init__(self, name: str, in_experiment: Experiment):
        """Define the Method name and the experiment it belongs to.
        Args:
            name: Name of method
            in_experiment: Part of experiment

        """
        self.name = name
        self.in_experiment = in_experiment
