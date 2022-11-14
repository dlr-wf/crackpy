import os

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from crackpy.crack_detection.deep_learning import nets
from crackpy.crack_detection.utils.utilityfunctions import calculate_segmentation


class UNetWithHooks(nets.UNet):
    """UNet with hooks registered during forward pass.
    This is done to save the gradients during backpropagation.
    Features can be registered and are then saved as well during forward pass.

    Methods:
        * forward - forward pass along with registration of backward gradient hooks
        * gradients_to_dict - save gradients to 'gradients_dict'
        * clear_grads_list - clear the gradient list and dict of the model
        * register_features - add forward hook after module[name] to catch features in forward pass
        * clear_feature_modules - clear feature modules of the model

    """

    def __init__(self, in_ch: int = 2, out_ch: int = 1, init_features: int = 64, dropout_prob: float = 0.0):
        """Initialize class arguments.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            init_features: number of initial features
            dropout_prob: dropout probability between 0 and 1

        """
        super().__init__(in_ch, out_ch, init_features, dropout_prob)

        self.feature_modules = {}
        self.features = {}

        self.gradients_list = []
        self.gradients = {}

    def forward(self, x):
        """Forward pass along with registration of backward gradient hooks.

        Args:
            x: input of UNet

        Returns:
            output before final sigmoid activation

        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        _ = x2.register_hook(self._save_gradients)  # gradient hook
        x3 = self.down2(x2)
        _ = x3.register_hook(self._save_gradients)  # gradient hook
        x4 = self.down3(x3)
        _ = x4.register_hook(self._save_gradients)  # gradient hook
        x5 = self.down4(x4)
        _ = x5.register_hook(self._save_gradients)  # gradient hook

        x6 = self.base(x5)
        _ = x6.register_hook(self._save_gradients)  # gradient hook

        x = self.up1(x6, x4)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.up2(x, x3)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.up3(x, x2)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.up4(x, x1)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.outc(x)

        return x  # no sigmoid!

    def gradients_to_dict(self):
        """Save gradients from 'gradients_list' to 'gradients_dict'."""
        names = ['up4', 'up3', 'up2', 'up1', 'base', 'down4', 'down3', 'down2', 'down1']
        for i, name in enumerate(names):
            self.gradients[name] = self.gradients_list[i]

    def clear_grad_list(self):
        """Clear the gradient list and dict of the model."""
        self.gradients_list = []
        self.gradients = {}

    def register_features(self, names):
        """Add forward hook after module[name] to catch features in forward pass.

        Args:
            names: name of the module

        """
        for name in names:
            # register forward hooks at feature modules
            m = dict(self.named_modules())[name]
            self.feature_modules[name] = m
            _ = m.register_forward_hook(self._save_features)

    def clear_feature_modules(self):
        """Clear the feature modules of the model."""
        self.feature_modules = {}

    def _save_gradients(self, grad):
        self.gradients_list.append(grad.data.numpy())

    def _save_features(self, m, i, output):
        self.features[m] = output.data.numpy()


class ParallelNetsWithHooks(nets.ParallelNets):
    """ParallelNets with hooks registered during forward pass of UNet.
    This is done to save the gradients during backpropagation.
    Features can be registered and are then saved as well during forward pass.

    Methods:
        * forward - forward pass along with registration of backward gradient hooks

    """

    def __init__(self, in_ch: int = 2, out_ch: int = 1, init_features: int = 64, dropout_prob: float = 0.0):
        """Initialize class arguments.

        Args:
            in_ch: number of input channels
            out_ch: number of output channels
            init_features: number of input features
            dropout_prob: dropout probability between 0 and 1

        """
        super().__init__(in_ch, out_ch, init_features, dropout_prob)

        self.unet = UNetWithHooks(in_ch=in_ch, out_ch=out_ch,
                                  init_features=init_features, dropout_prob=dropout_prob)

    def forward(self, x):
        """Forward pass along with registration of backward gradient hooks.

        Args:
            x: input of ParallelNets

        Returns:
            output before final sigmoid activation

        """
        self.unet(x)
        return x


class SegGradCAM:
    """This class implements Grad-CAM for semantic segmentation tasks as described in
    Natekar et al. 2020 (see https://doi.org/10.3389/fncom.2020.00006)
    Vinogradova et al. (see https://arxiv.org/pdf/2002.11434.pdf)

    Methods:
        * plot - plot the model's Seg-Grad-CAM heatmap together with the model's segmentation output
        * save - save figure of stage

    """

    def __init__(self, setup, model, feature_modules: list = None, mask: bool = False):
        """Initialize class arguments.

        Args:
            setup: (Setup-Class object) from crackpy.crack_detection.deep_learning.setup
            model: (Pytorch model) e.g. UNet, ParallelNets
            feature_modules: list[str] e.g. ['down4', 'base', 'up1'] or None
            mask: if True, a segmentation mask is multiplied by the output (see Vinogradova et al.)

        """
        self.setup = setup
        self.model = model
        self.mask = mask

        if feature_modules is None:
            # use feature modules provided by 'setup'
            layers = self.setup.visu_layers
            if layers is not None:
                self.feature_modules = layers
            else:
                raise ValueError("No visualization layers specified in setup.")

        else:
            self.feature_modules = feature_modules

        if isinstance(self.feature_modules, str):
            # make list out of single feature module
            self.feature_modules = [self.feature_modules]

        self.model.register_features(self.feature_modules)

    def __call__(self, input_t: torch.Tensor):
        """Calculate Grad-CAM for segmentation model.

        Args:
            input_t: input for which the CAM heatmap should be calculated

        """
        self.model.eval()

        # forward pass
        output = self.model(input_t)

        ones = torch.ones(output.size()).requires_grad_(True)
        if self.mask:
            # build mask
            is_seg = torch.BoolTensor(output >= 0)
            mask = torch.where(is_seg, 1, 0)
            mask = mask * ones
        else:
            # fill with ones
            mask = ones

        #########################################################################################
        # global (average) pooling of the output
        # out = torch.mean(mask * output)  # Natekar et al. 2020 <-- no difference between these!
        out = torch.sum(mask * output)  # Vinogradova et al. 2020 <-- nicer scale!
        #########################################################################################

        # backpropagation
        self.model.zero_grad()
        self.model.clear_grad_list()
        out.backward(retain_graph=True)
        self.model.gradients_to_dict()

        # calculate activation mapping
        seg_cam = self._calculate_cam(size=input_t.shape[-2:])

        return output, seg_cam

    def _calculate_cam(self, size):
        cam = np.zeros(size, dtype=np.float32)

        for name in self.feature_modules:

            m = self.model.feature_modules[name]
            feats = self.model.features[m]
            grads = self.model.gradients[name]
            weights = np.mean(grads, axis=(-2, -1))[0, :]  # global average pooling of gradients
            ############################################################################
            # weights = np.maximum(0, weights)  # uncomment for taking relu on gradients
            ############################################################################
            current_cam = np.zeros(feats.shape[-2:], dtype=np.float32)
            for i, weight in enumerate(weights):
                current_cam += weight * feats[0, i, :, :]

            cam += cv2.resize(current_cam, size)

        # clip negative values --> highlight only areas that positively contribute to segmentation
        cam = np.maximum(cam, 0)

        # flip left<->right (if side is left)
        if self.setup.side == 'left':
            cam = np.fliplr(cam)

        return cam

    def plot(self,
             output: torch.Tensor,
             heatmap: np.array,
             scale: str = 'QUALITATIVE',
             show: bool = False
             ) -> plt.figure:
        """Plot the model's Seg-Grad-CAM heatmap together with the model's segmentation output.

        Args:
            output: output of model or application of Seg-Grad-CAM-forward
            heatmap: Seg-Grad-CAM attention heatmap
            scale:'QUALITATIVE' or 'QUANTITATIVE'
            show: whether to display the plot

       Returns:
            plt.figure of the plot

        """
        # setup color vectors and normalize heatmap if necessary
        num_colors = 120

        if scale == 'QUALITATIVE':
            # normalize heatmaps to [0, 1]
            heatmap = heatmap - np.nanmin(heatmap)
            if np.nanmax(heatmap) != 0:
                heatmap = heatmap / np.nanmax(heatmap)
            # set contour and label vector
            contour_vector = np.linspace(0, 1, num_colors)
            label_vector = ['Low', 'High']

        elif scale == 'QUANTITATIVE':
            contour_vector = np.linspace(np.nanmin(heatmap), np.nanmax(heatmap), num_colors)
            label_vector = np.linspace(np.nanmin(heatmap), np.nanmax(heatmap), 2)

        else:
            raise AssertionError("Parameter 'scale' must be 'QUALITATIVE' for qualitative/percent "
                                 "scale or 'QUANTITATIVE' for quantitative scale of heatmaps")
        plt.clf()

        # initialize figure
        fig = plt.figure(1, figsize=(12, 9))
        ax = fig.add_subplot(111)

        # get coordinates to interpolate heatmap on
        pixels = heatmap.shape[0]

        if self.setup.side == 'right':
            interp_coor_x = np.linspace(self.setup.offset[0], self.setup.size + self.setup.offset[0], pixels)
            interp_coor_y = np.linspace(-self.setup.size / 2.0 + self.setup.offset[1],
                                        self.setup.size / 2.0 + self.setup.offset[1], pixels)
        else:
            interp_coor_x = np.linspace(-self.setup.size + self.setup.offset[0], self.setup.offset[0], pixels)
            interp_coor_y = np.linspace(-self.setup.size / 2.0 + self.setup.offset[1],
                                        self.setup.size / 2.0 + self.setup.offset[1], pixels)

        coor_x, coor_y = np.meshgrid(interp_coor_x, interp_coor_y)

        # plot heatmap
        triang = tri.Triangulation(coor_x.flatten(), coor_y.flatten())
        mask = np.any(np.where(np.isnan(heatmap.flatten())[triang.triangles], True, False),
                      axis=1)
        triang.set_mask(mask)
        plot = ax.tricontourf(triang,
                              heatmap.flatten(), contour_vector,
                              extend='neither', cmap='jet')
        ax.autoscale(False)
        # ax.axis('off')  # uncomment to turn off axis labels and ticks

        # calculate crack tip segmentation
        tip_seg = calculate_segmentation(torch.sigmoid(output))
        signed_size = self.setup.size if self.setup.side == 'right' else -self.setup.size
        x_tip_seg = tip_seg[:, 1] * signed_size / pixels + self.setup.offset[0]
        y_tip_seg = tip_seg[:, 0] * self.setup.size / pixels - self.setup.size / 2 + self.setup.offset[1]
        # plot crack tip segmentation
        ax.scatter(x_tip_seg, y_tip_seg, color='grey', alpha=0.5, linewidths=1, marker='.')

        # plot color bar and setup axis labels and ticks
        cbar = fig.colorbar(plot, ticks=[0, 1], label='Attention', format='%.0f')
        cbar.ax.set_yticklabels(label_vector)
        # ax.set_title(f'{key}')  # comment to turn off title
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_xlim(coor_x.min(), coor_x.max())
        ax.set_ylim(coor_y.min(), coor_y.max())

        if show:
            plt.show()

        return fig

    def save(self, key, fig: plt.figure, subfolder: str = None):
        """Save figure 'fig' of stage 'key' in folder 'subfolder'.

        Args:
            key: nodemap key
            fig: plt.figure of heatmap
            subfolder: name of subfolder where the figue should be saved

        """
        stage_num = self.setup.nodemaps_to_stages[key]
        save_folder = os.path.join(self.setup.output_path)
        if subfolder is not None:
            save_folder = os.path.join(save_folder, subfolder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, f'{stage_num:04d}' + '.png'))
        plt.close(fig)


def plot_overview(output, maps, side: str, scale: str = 'QUALITATIVE'):
    """Plot Seg-Grad-CAM of several layers in one single figure.

    Args:
        output: model output
        maps: heatmaps and arrays
        side: side of the specimen, e.g. 'right'
        scale: 'QUALITATIVE' for scaling each heatmap to [0,1]
               'QUANTITATIVE' skip scaling

    """
    # setup color bar settings and scaling of heatmaps
    num_colors = 120

    if scale == 'QUALITATIVE':
        # normalize heatmaps to [0, 1]
        maps_scaled = {}
        for key, cam in maps.items():
            cam = cam - np.nanmin(cam)
            if cam.max() != 0:
                cam = cam / np.nanmax(cam)
            maps_scaled[key] = cam
        # set contour and label vector
        f_min, f_max = 0, 1
        contour_vector = np.linspace(f_min, f_max, num_colors)
        label_vector = ['Low', 'High']

    elif scale == 'QUANTITATIVE':
        # calculate min and max along all heatmaps
        heatmaps_list = []
        for heatmap in maps.values():
            heatmaps_list.append(heatmap)
        heatmaps_array = np.asarray(heatmaps_list)
        f_min = np.nanmin(heatmaps_array)
        f_max = np.nanmax(heatmaps_array)
        if f_max - f_min == 0:
            f_max = f_min + 1
        maps_scaled = maps
        # set contour and label vector
        contour_vector = np.linspace(f_min, f_max, num_colors)
        label_vector = [f'{int(f_min)}', f'{int(f_max)}']

    else:
        raise AssertionError("Parameter 'scale' must be 'QUALITATIVE' for qualitative/percent "
                             "scale or 'QUANTITATIVE' for quantitative scale of heatmaps")
    plt.clf()

    # get coordinates to interpolate heatmap on
    pixels = next(iter(maps.values())).shape[-1]
    interp_coor_x = np.linspace(0, pixels, pixels)
    interp_coor_y = np.linspace(0, pixels, pixels)
    coor_x, coor_y = np.meshgrid(interp_coor_x, interp_coor_y)

    # initialize figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))

    for ax, (name, heatmap) in zip(axs.flat, maps_scaled.items()):
        ax.axis('off')
        ax.set_title(f'{name}')

        # plot heatmap
        plot = ax.tricontourf(coor_x.flatten(), coor_y.flatten(),
                              heatmap.flatten(), contour_vector,
                              extend='neither', cmap='jet')
        ax.autoscale(False)

        # calculate crack tip segmentation
        tip_seg = calculate_segmentation(torch.sigmoid(output))
        if side == 'left':
            tip_seg[:, 1] = pixels - tip_seg[:, 1]

        # plot crack tip segmentation
        ax.scatter(tip_seg[:, 1], tip_seg[:, 0], color='grey', alpha=0.5, linewidths=1, marker='.')

    # plot color bar
    cbar = fig.colorbar(plot, ax=axs, ticks=[f_min, f_max], label='Attention')
    cbar.ax.set_yticklabels(label_vector)

    return fig
