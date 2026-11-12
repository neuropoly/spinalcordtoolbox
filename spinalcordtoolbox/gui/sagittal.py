"""
Qt dialog for manually segmenting a spinalcord image

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging

import numpy as np
from PyQt5 import QtWidgets

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import TooManyPointsWarning, MissingLabelWarning

logger = logging.getLogger(__name__)


class SagittalController(base.BaseController):
    def __init__(self, image, params, init_values=None, previous_point=None):
        super(SagittalController, self).__init__(image, params, init_values)

        if previous_point is not None:
            for i in range(len(previous_point)):
                self.points.append(previous_point[i])

    def select_point(self, x, y, z, label):
        if not self.valid_point(x, y, z):
            raise ValueError(f'Invalid coordinates {(x, y, z)}')

        existing_point = [i for i, j in enumerate(self.points) if j[3] == label]

        if existing_point:
            self.points[existing_point[0]] = (x, y, z, label)
        else:
            if self.params.num_points and len(self.points) >= self.params.num_points:
                raise TooManyPointsWarning()
            self.points.append((x, y, z, label))

        self.position = (x, y, z)


class SagittalDialog(base.BaseDialog):

    def _init_canvas(self, parent):
        layout = QtWidgets.QHBoxLayout()
        parent.addLayout(layout)

        self.labels = widgets.VertebraeWidget(self, self.params.vertebraes)
        self.labels.label = self._controller.label = self.params.start_vertebrae
        layout.addWidget(self.labels)

        self.sagittal = widgets.SagittalCanvas(self, plot_points=True, annotate=True)
        self.sagittal.title(self.params.subtitle)
        self.sagittal.point_selected_signal.connect(self.on_select_point)
        layout.addWidget(self.sagittal)
        self.sagittal.refresh()

    def _init_controls(self, parent):
        pass

    def on_select_point(self, x, y, z):
        try:
            x, y, z = np.array(np.round((x, y, z)), dtype=int)
            label = self.labels.label
            self._controller.select_point(x, y, z, label)
            self.labels.refresh()
            self.sagittal.refresh()

            index = self.params.vertebraes.index(label)
            if index + 1 < len(self.params.vertebraes):
                self.labels.label = self.params.vertebraes[index + 1]
        except (TooManyPointsWarning, MissingLabelWarning) as warn:
            self.update_warning(str(warn))

    def on_undo(self):
        super(SagittalDialog, self).on_undo()
        self.sagittal.refresh()
        self.labels.refresh()
        self.labels.label = self._controller.label

    def increment_vertical_nav(self):
        x, y, z = self._controller.position
        z_bound = self._controller.image.dim[2]
        z_ = z + 1 if (z + 1) < z_bound else -z_bound
        self._controller.position = (x, y, z_)
        self.sagittal.refresh()

    def decrement_vertical_nav(self):
        x, y, z = self._controller.position
        z_bound = self._controller.image.dim[2]
        z_ = z - 1 if (z - 1) >= -z_bound else z_bound - 1
        self._controller.position = (x, y, z_)
        self.sagittal.refresh()


def launch_sagittal_dialog(input_file, output_file, params, previous_points=None):
    if not params.vertebraes:
        params.vertebraes = [3, 5]
    params.input_file_name = input_file.absolutepath
    params.subtitle += ("[KEYBOARD] Left/Right arrows: Navigate across slices."
                        "\n[MOUSE] Right click: Change brightness (left/right) and contrast (up/down)."
                        "\n[MOUSE] Scrolling middle button: Zoom in/out.")
    controller = SagittalController(input_file, params, output_file, previous_points)
    controller.reformat_image()

    app = QtWidgets.QApplication([])
    dialog = SagittalDialog(controller)
    dialog.show()
    app.exec_()
    return controller
