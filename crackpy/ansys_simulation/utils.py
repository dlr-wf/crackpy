import os


def delete_ansys_leftovers(ansys_folder: str):
    """Delete overhead Ansys files after the run.

    Args:
        ansys_folder: path of ansys run location

    """

    files = os.listdir(ansys_folder)
    keep_endings = ('.txt', '.png', '.vtk', '.pdf', '.eps', '.svg')
    for ansys_file in files:
        if not ansys_file.endswith(keep_endings):
            os.remove(os.path.join(ansys_folder, ansys_file))
