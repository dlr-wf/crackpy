import numpy as np
from scipy.interpolate import griddata

from crackpy.fracture_analysis.data_processing import InputData


def interpolate_on_array(input_by_nodemap, interp_size, offset=(0, 0), pixels=256):
    """Extracts Coors, Disps and EpsVonMises from dictionary of input data objects.
    Interpolates these quantities onto an array of a certain pixel size by calling
    the function InterpOnArray for each entry in dictionary.

    Args:
        input_by_nodemap: (dict of InputData objects)
        interp_size: indicates the edge length of field of view
                    This field of view is located at 0 < x < size and -size/2 < y < size/2
        offset: indicates the x-value offset (can be negative)
        pixels: indicates the edge length of field of view in pixels

    Returns:
        (dicts) of coordinates, displacements and EpsVonMises interpolated on arrays

    """
    print('\rData will be interpolated on an array...')

    interp_coors_by_nodemap = {}
    interp_disps_by_nodemap = {}
    interp_eps_vm_by_nodemap = {}

    for key in input_by_nodemap:
        print(f'\r- {key}. {len(interp_disps_by_nodemap.keys()) + 1}/{len(input_by_nodemap.keys())} interpolated.',
              end='')

        interp_coors, interp_disps, interp_eps_vm = interpolate(input_by_nodemap[key],
                                                                interp_size,
                                                                pixels=pixels,
                                                                offset=offset)

        interp_coors_by_nodemap.update({key: interp_coors})
        interp_disps_by_nodemap.update({key: interp_disps})
        interp_eps_vm_by_nodemap.update({key: interp_eps_vm})

    print('\n')

    return interp_coors_by_nodemap, interp_disps_by_nodemap, interp_eps_vm_by_nodemap


def interpolate(input_data_object: InputData, size: int, offset=(0, 0), pixels: int=256):
    """Extracts and interpolates Coors, Disps and EpsVonMises
    from an InputData object onto arrays of size pixels x pixels.

    Args:
        input_data_object: This object gives access to all DIC data for a given nodemap
                        This can be accessed by e.g. InputData.dXVec for the X-Vector of DIC data
        size: indicates the edge length of field of view.
                This field of view is located at 0 < x < size and -size/2 < y < size/2
        offset: (x,y)-offset tuple
        pixels: indicates the edge length of field of view in pixels

    Returns:
        Arrays interp_coors, interp_disps, interp_eps_vm all with a shape (pixels, pixels)
        giving coordinates, displacements and Von-Mises Strains on each pixel in the array

    """
    if not isinstance(size, (int, float)):
        raise TypeError(
            'Argument "size" should be an integer, indicating the edge length of field of view.\n'
            'This field of view is located at 0 < x < size and -size/2 < y < size/2')
    if not isinstance(pixels, int):
        raise TypeError(
            'Argument "pixels" should be an integer, indicating the edge length of field of view '
            'in pixels.\n Default is 256')

    offset_x, offset_y = offset

    x_coordinate = input_data_object.coor_x
    y_coordinate = input_data_object.coor_y

    x_displacement = input_data_object.disp_x
    y_displacement = input_data_object.disp_y

    eps_vm = input_data_object.eps_vm

    if size >= 0:
        x_coor_interp = np.linspace(offset_x, size + offset_x, pixels)
        y_coor_interp = np.linspace(-size / 2.0 + offset_y, size / 2.0 + offset_y, pixels)
    else:
        x_coor_interp = np.linspace(size + offset_x, offset_x, pixels)
        y_coor_interp = np.linspace(size / 2.0 + offset_y, -size / 2.0 + offset_y, pixels)

    x_grid, y_grid = np.meshgrid(x_coor_interp, y_coor_interp)

    x_disp_interp = griddata((x_coordinate, y_coordinate), x_displacement, (x_grid, y_grid))
    y_disp_interp = griddata((x_coordinate, y_coordinate), y_displacement, (x_grid, y_grid))

    interp_eps_vm = griddata((x_coordinate, y_coordinate), eps_vm, (x_grid, y_grid))

    # If size is negative, then the left-hand side of the specimen is flipped such that the crack
    # always starts from the left and the x-displacement is multiplied by -1
    if size < 0:
        x_disp_interp = np.fliplr(x_disp_interp) * -1
        y_disp_interp = np.fliplr(y_disp_interp)
        interp_eps_vm = np.fliplr(interp_eps_vm)
        x_grid = np.fliplr(x_grid)
        y_grid = np.fliplr(y_grid)

    interp_coors = np.asarray([x_grid, y_grid])
    interp_disps = np.asarray([x_disp_interp, y_disp_interp])

    return interp_coors, interp_disps, interp_eps_vm
