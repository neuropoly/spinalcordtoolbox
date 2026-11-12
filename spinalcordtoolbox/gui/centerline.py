"""
Qt dialog for manual labeling of an image

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
import sys

import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui

from spinalcordtoolbox.gui import base
from spinalcordtoolbox.gui import widgets
from spinalcordtoolbox.gui.base import InvalidActionWarning

# TODO: remove this useless logger (because no handler is found) by sct.log
logger = logging.getLogger(__name__)


class CenterlineController(base.BaseController):
    _mode = ''
    START_SLICE = 0
    INTERVAL = 15
    MODES = ['AUTO', 'CUSTOM']

    def __init__(self, image, params, init_values=None):
        super(CenterlineController, self).__init__(image, params, init_values)

    def reformat_image(self):
        # reorient data to SAL
        super(CenterlineController, self).reformat_image()
        max_x, max_z = self.image.dim[:3:2]
        # self.params.num_points = self.params.num_points or 11
        # update interval (in pixel) between two consecutive points based on pixel size
        self.INTERVAL = np.round(self.params.interval_in_mm // self.image.dim[4])

        # set first slice location (see definitions in base.py)
        if self.params.starting_slice == 'fov':
            self.START_SLICE = 0
        elif self.params.starting_slice == 'top_minus_one':
            self.START_SLICE = 1
        elif self.params.starting_slice == 'midfovminusinterval':
            self.START_SLICE = np.round(self.image.dim[0] / 2 - self.INTERVAL)

        # if the starting slice is of invalid value then use default value
        if self.START_SLICE > max_x or self.START_SLICE < 0:
            self.START_SLICE = self.default_position[0]
            logger.warning('Starting slice value is out of range')
        # if the starting slice is a fraction, recalculate the starting slice as a ratio.
        elif 0 < self.START_SLICE < 1:
            self.START_SLICE = max_z // self.START_SLICE

        self.reset_position()

    def reset_position(self):
        super(CenterlineController, self).reset_position()
        self.position = (self.START_SLICE, self.position[1], self.position[2])

    def skip_slice(self):
        if self.mode == 'AUTO':
            self.increment_slice()

    def select_point(self, x, y, z):
        if not self.valid_point(x, y, z):
            raise ValueError(f'Invalid point selected {(x, y, z)}')

        logger.debug(f'Point Selected {(x, y, z)}')

        existing_points = [i for i, j in enumerate(self.points) if j[0] == x]
        if existing_points:
            self.points[existing_points[0]] = (x, y, z, 1)
        else:
            self.points.append((x, y, z, 1))
        self.position = (x, y, z)
        if self.mode == 'AUTO':
            self.increment_slice()

    def increment_slice(self):
        interval = 1
        if self.mode == 'AUTO':
            interval = self.INTERVAL

        x, y, z = self.position
        new_x = x + interval
        max_x = self.image.dim[0]

        if new_x >= max_x:
            new_x = max_x - 1

        if self.valid_point(new_x, y, z):
            self.position = (new_x, y, z)

    def decrement_slice(self):
        interval = 1
        if self.mode == 'AUTO':
            interval = self.INTERVAL

        x, y, z = self.position
        new_x = x - interval

        if new_x < 0:
            new_x = 0

        if self.valid_point(new_x, y, z):
            self.position = (new_x, y, z)

    def select_slice(self, x, y, z):
        if self.mode != 'CUSTOM':
            raise InvalidActionWarning('Can only select a slice in CUSTOM mode')

        if not self.valid_point(x, y, z):
            raise ValueError(f'Invalid slice selected {(x, y, z)}')

        _, y, z = self.position
        logger.debug(f'Slice Selected {(x, y, z)}')
        self.position = (x, y, z)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value not in self.MODES:
            raise ValueError(f'Invalid mode {value}')

        if value != self._mode:
            self._mode = value


class Centerline(base.BaseDialog):
    def __init__(self, *args, **kwargs):
        super(Centerline, self).__init__(*args, **kwargs)
        self.axial_canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    def _init_canvas(self, parent):
        layout = QtWidgets.QHBoxLayout()
        parent.addLayout(layout)
        self.sagittal_canvas = widgets.SagittalCanvas(self, plot_points=True, horizontal_nav=True)
        self.sagittal_canvas.title(self.params.subtitle)
        layout.addWidget(self.sagittal_canvas)

        self.axial_canvas = widgets.AxialCanvas(self, plot_points=True, crosshair=True)
        self.axial_canvas.plot_points()
        layout.addWidget(self.axial_canvas)

        self.axial_canvas.point_selected_signal.connect(self.on_select_point)
        self.sagittal_canvas.point_selected_signal.connect(self.on_select_slice)

    def _init_controls(self, parent):
        group = QtWidgets.QGroupBox()
        group.setFlat(True)
        layout = QtWidgets.QHBoxLayout()

        if sys.platform.startswith('darwin'):
            cmd_key = 'Cmd'
        else:
            cmd_key = 'Ctrl'

        custom_mode = QtWidgets.QRadioButton(f'Mode Custom [{cmd_key}+T]')
        custom_mode.mode = 'CUSTOM'
        custom_mode.setToolTip('Manually select the axis slice on sagittal plane')
        custom_mode.toggled.connect(self.on_toggle_mode)
        custom_mode.sagittal_title = f'Select an axial slice.\n{self.params.subtitle}'
        custom_mode.axial_title = 'Click in the center of the spinal cord'
        layout.addWidget(custom_mode)

        auto_mode = QtWidgets.QRadioButton(f'Mode Auto [{cmd_key}+T]')
        auto_mode.mode = 'AUTO'
        auto_mode.setToolTip('Automatically move down the axis slice on the sagittal plane')
        auto_mode.toggled.connect(self.on_toggle_mode)
        auto_mode.sagittal_title = f'The axial slice is automatically selected\n{self.params.subtitle}'
        auto_mode.axial_title = 'Click in the center of the spinal cord'
        layout.addWidget(auto_mode)

        group.setLayout(layout)
        parent.addWidget(group)
        auto_mode.click()

        # Add keyboard shortcut to toggle between modes
        self.auto_mode = auto_mode
        self.custom_mode = custom_mode
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+T'), self, self.on_toggle_mode)

    def _init_footer(self, parent):
        ctrl_layout = super(Centerline, self)._init_footer(parent)

        if sys.platform.startswith('darwin'):
            cmd_key = 'Cmd'
        else:
            cmd_key = 'Ctrl'

        skip = QtWidgets.QPushButton(f'Skip [{cmd_key}+F]')
        ctrl_layout.insertWidget(2, skip)
        skip.clicked.connect(self.on_skip_label)

        clean = QtWidgets.QPushButton(f'Delete all [{cmd_key}+D]')
        ctrl_layout.insertWidget(2, clean)
        clean.clicked.connect(self.on_delete_all_labels)
        # TODO: try the setShortcut attribute

        events = (
            (QtGui.QKeySequence('Ctrl+F'), self.on_skip_label),
            (QtGui.QKeySequence('Ctrl+D'), self.on_delete_all_labels),
        )
        for event, action in events:
            QtWidgets.QShortcut(event, self, action)

    def on_delete_all_labels(self):
        try:
            logger.debug("Clean all labels")
            self._controller.points = []
            self.sagittal_canvas.refresh()
            self.axial_canvas.refresh()
            # TODO: fix: does not refresh
            # TODO: add text
        except InvalidActionWarning as warn:
            self.update_warning(str(warn))

    def on_skip_label(self):
        try:
            logger.debug('Skipping slice')
            self._controller.skip_slice()
            self.sagittal_canvas.refresh()
            self.axial_canvas.refresh()
        except InvalidActionWarning as warn:
            self.update_warning(str(warn))

    def on_toggle_mode(self):
        widget = self.sender()
        # If needed in case user used keyboard shortcut (no widget.mode in this case)
        if hasattr(widget, 'mode'):
            if widget.mode == self._controller.mode:
                # User clicked on the same button, so no further action is required.
                return
        else:
            # If user selected keyboard shortcut, simply toggle mode without further checks.
            if self._controller.mode == 'AUTO':
                widget = self.custom_mode
            else:
                widget = self.auto_mode
            widget.click()
        self._controller.mode = widget.mode
        self.update_status(f'Now in mode {widget.mode}')
        self.sagittal_canvas.title(widget.sagittal_title)
        self.sagittal_canvas.refresh()
        self.axial_canvas.title(widget.axial_title)
        self.axial_canvas.refresh()

    def on_select_slice(self, x, y, z):
        try:
            logger.debug(f'Slice clicked {(x, y, z)}')
            self._controller.select_slice(x, y, z)
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status(f'Sagittal position at {self._controller.position[0]:8.2f}')
        except InvalidActionWarning as warn:
            self.update_warning(str(warn))

    def increment_horizontal_nav(self):
        self._controller.increment_slice()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()

    def decrement_horizontal_nav(self):
        self._controller.decrement_slice()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()

    def on_select_point(self, x, y, z):
        try:
            logger.debug(f'Point clicked {(x, y, z)}')
            self._controller.select_point(x, y, z)
            self.axial_canvas.refresh()
            self.sagittal_canvas.refresh()
            self.update_status(f'{len(self._controller.points)} point(s) selected')
        except InvalidActionWarning as warn:
            self.update_warning(str(warn))

    def on_undo(self):
        super(Centerline, self).on_undo()
        self.axial_canvas.refresh()
        self.sagittal_canvas.refresh()


def launch_centerline_dialog(im_input_orig, im_output, params):
    """
    Launch GUI where user clicks on axial view to generate centerline information.
    :param im_input_orig: input image object
    :param im_output: output image object that will contain the generated manual labels
    :param params: # TODO: define all params
    :return:
    """
    im_input = im_input_orig.copy()  # need to copy otherwise the field absolutepath becomes None when exiting
    params.input_file_name = im_input.absolutepath
    params.subtitle += ("[KEYBOARD] Up/Down arrows: Navigate the superior-inferior direction"
                        "\n[MOUSE] Right click: Change brightness (left/right) and contrast (up/down)."
                        "\n[MOUSE] Scrolling middle button: Zoom in/out.")
    controller = CenterlineController(im_input, params, im_output)
    controller.reformat_image()

    app = QtWidgets.QApplication([])
    dialog_ = Centerline(controller)
    dialog_.show()
    app.exec_()
    return controller
