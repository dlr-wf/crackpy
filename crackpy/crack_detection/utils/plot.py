import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri, cm
from matplotlib.colors import ListedColormap


def plot_prediction(
        background: np.array,
        interp_size: float or int,
        offset: tuple=(0, 0),
        crack_tip_prediction: np.array=None,
        crack_tip_seg: np.array=None,
        crack_tip_label: np.array=None,
        crack_path: np.array=None,
        f_min: float=None,
        f_max: float=None,
        save_name: str=None,
        path: str=None,
        title: str='',
        label: str='Plot of Data [Unit]'
):
    """Plots crack tip labels and predictions over background.

    Args:
        background: for background data
        interp_size: actual size of background (can be negative!)
        offset: indicates the x-value offset (can be negative!)
        crack_tip_prediction: array of size 1 x 2 - [:, 0] are y-coordinates!
        crack_tip_seg: array of size num_of_segmentations x 2
        crack_tip_label: array of size 1 x 2 - [:, 0] are y-coordinates!
        crack_path: array of size H x W
        f_min: minimal value of background data in plot (if None then auto-min)
        f_max: maximal value of background data in plot (if None then auto-max)
        save_name: Name under which the plot is saved (.png added automatically)
        path: location to save plot
        title: of the plot
        label: label for background data color bar

    """
    if f_min is None:
        f_min = background.min()
    if f_max is None:
        f_max = background.max()
        extend = 'neither'
    else:
        extend = 'max'

    num_colors = 120
    contour_vector = np.linspace(f_min, f_max, num_colors, endpoint=True)
    label_vector = np.linspace(f_min, f_max, 10, endpoint=True)
    # Colormap similar to Aramis
    cm_jet = cm.get_cmap('jet', 512)
    my_cmap = ListedColormap(cm_jet(np.linspace(0.1, 0.9, 256)))

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    pixels = background.shape[0]

    if interp_size >= 0:
        x_coor_interp = np.linspace(offset[0], interp_size + offset[0], pixels)
        y_coor_interp = np.linspace(-interp_size / 2.0 + offset[1], interp_size / 2.0 + offset[1], pixels)
    else:
        x_coor_interp = np.linspace(interp_size + offset[0], offset[0], pixels)
        y_coor_interp = np.linspace(interp_size / 2.0 + offset[1], -interp_size / 2.0 + offset[1], pixels)
        background = np.fliplr(background)

    coor_x, coor_y = np.meshgrid(x_coor_interp, y_coor_interp)
    triang = tri.Triangulation(coor_x.flatten(), coor_y.flatten())

    mask = np.any(np.where(np.isnan(background.flatten())[triang.triangles], True, False), axis=1)
    triang.set_mask(mask)
    plot = ax.tricontourf(triang,
                          background.flatten(), contour_vector,
                          extend=extend, cmap=my_cmap)
    ax.autoscale(False)
    # ax.axis('off')  # uncomment to turn of axis and labels

    size = np.abs(interp_size)
    if crack_tip_seg is not None:
        # crack tip segmentation
        x_crack_tip_seg = crack_tip_seg[:, 1] * interp_size / (pixels - 1) + offset[0]
        y_crack_tip_seg = crack_tip_seg[:, 0] * size / (pixels - 1) - size / 2 + offset[1]
        ax.scatter(x_crack_tip_seg, y_crack_tip_seg, color='gray', linewidths=1, marker='.')

    if crack_path is not None:
        # crack path segmentation
        x_crack_path = crack_path[:, 1] * interp_size / (pixels - 1) + offset[0]
        y_crack_path = crack_path[:, 0] * size / (pixels - 1) - size / 2 + offset[1]
        ax.scatter(x_crack_path, y_crack_path, color='black', s=1, marker='.')

    if crack_tip_label is not None:
        # actual crack tip label
        x_crack_tip_label = crack_tip_label[:, 1] * interp_size / (pixels - 1) + offset[0]
        y_crack_tip_label = crack_tip_label[:, 0] * size / (pixels - 1) - size / 2 + offset[1]
        ax.scatter(x_crack_tip_label, y_crack_tip_label, color='black', linewidths=1, marker='x')

    if crack_tip_prediction is not None:
        # crack tip prediction
        x_crack_tip_pred = crack_tip_prediction[:, 1] * interp_size / (pixels - 1) + offset[0]
        y_crack_tip_pred = crack_tip_prediction[:, 0] * size / (pixels - 1) - size / 2 + offset[1]
        ax.scatter(x_crack_tip_pred, y_crack_tip_pred, color='darkred', linewidths=1, marker='x')

    fig.colorbar(plot, ticks=label_vector, label=label)
    ax.set_title(title)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')

    ax.set_xlim(coor_x.min(), coor_x.max())
    ax.set_ylim(coor_y.min(), coor_y.max())

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, save_name + '.png'), bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()
