"""
Base classes for creating GUI objects to create manually selected points

The definition of X,Y axis is the following:

  xmin,ymin o---------o xmax,ymin
            |         |
            |         |
            |         |
            |         |
  xmin,ymax o---------o xmax,ymax

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

from collections import namedtuple

import logging
import sys

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)


Position = namedtuple('Position', ('x', 'y', 'z'))


class AnatomicalParams(object):
    """The base parameter object for GUI configuration"""
    def __init__(self,
                 cmap='gray',
                 interp='nearest',
                 perc_min=5.,
                 perc_max=95.,
                 vmode='percentile',
                 alpha=1.0):
        """

        Parameters
        ----------
        cmap : str
        interp : str
        perc_min : float: low percentile threshold for intensity adjustment
        perc_max : float: high percentile threshold for intensity adjustment
        vmode : str: "percentile": intensity adjustment based on vmin/vmax percentile,
                     "mean-std": intensity adjustment based on
                     "clahe: CLAHE (not implemented yet)
        alpha : float
        """
        self.cmap = cmap
        self.interp = interp
        self.perc_min = perc_min
        self.perc_max = perc_max
        self.vmode = vmode
        self.alpha = alpha
        self.start_vertebrae = 50
        self.end_vertebrae = -1
        self.num_points = 0
        self._title = ''  # figure title
        self.subtitle = ''  # subplot title (will be displayed above the image)
        self._vertebraes = []
        self.input_file_name = ""
        self.starting_slice = 'top'  # used in centerline.py canvas and corresponds to the location of
        # the first axial slice for labeling. Possible values are: 'top': top slice; 'midfovminusinterval': mid-FOV
        # minus the interval.
        self.interval_in_mm = 15  # superior-inferior distance between two consecutive labels in AUTO mode

    @property
    def dialog_title(self):
        if not self._title:
            self._title = f'{self.input_file_name}: manual labeling'
        return self._title

    @property
    def vertebraes(self):
        return self._vertebraes

    @vertebraes.setter
    def vertebraes(self, values):
        if not values:
            return

        self._vertebraes = values
        self.start_vertebrae = values[0]
        self.end_vertebrae = values[-1]


class BaseDialog(QtWidgets.QWidget):
    """Abstract base class to a Anatomical GUI.

    Attributes
    ----------
    update_canvas_signal : QtCore.Signal
        Signal emits when dialog has a point to add to the

    """
    lb_status = None
    lb_warning = None
    btn_ok = None
    btn_undo = None

    def __init__(self, controller):
        """Initialize the UI parameters

        Parameters
        ----------
        controller : BaseController
            The logical object that controls the state of the UI
        """
        super(BaseDialog, self).__init__()
        self.params = controller.params
        self._controller = controller
        self.image = controller.image
        self._controller._dialog = self
        self._init_ui()

    def _init_ui(self):
        self.resize(1200, 800)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        layout = QtWidgets.QVBoxLayout(self)

        self._init_header(layout)
        self._init_canvas(layout)
        self._init_controls(layout)
        self._init_footer(layout)

        events = (
            (QtGui.QKeySequence.Undo, self.on_undo),
            (QtGui.QKeySequence.Save, self.on_save_quit),
            (QtGui.QKeySequence.Quit, self.close),
            (QtGui.QKeySequence.MoveToNextChar, self.increment_vertical_nav),
            (QtGui.QKeySequence.MoveToPreviousChar, self.decrement_vertical_nav),
            (QtGui.QKeySequence.MoveToNextLine, self.increment_horizontal_nav),
            (QtGui.QKeySequence.MoveToPreviousLine, self.decrement_horizontal_nav)
        )

        for event, action in events:
            QtWidgets.QShortcut(event, self, action)

        self.setWindowTitle(self.params.dialog_title)

    def increment_vertical_nav(self):
        """Action to increment the anatonical viewing position.

        The common case is when the right arrow key is pressed. Ignore implementing
        this function if no navigation functionality is required
        """
        pass

    def decrement_vertical_nav(self):
        """Action to decrement the anatonical viewing position.

        The common case is when the left arrow key is pressed. Ignore implementing
        this function if no navigation functionality is required
        """
        pass

    def increment_horizontal_nav(self):
        """Action to increment the anatonical viewing position.

        The common case is when the down arrow key is pressed. Ignore implementing
        this function if no navigation functionality is required
        """
        pass

    def decrement_horizontal_nav(self):
        """Action to decrement the anatonical viewing position.

        The common case is when the up arrow key is pressed. Ignore implementing
        this function if no navigation functionality is required
        """
        pass

    def _init_canvas(self, parent):
        """

        Parameters
        ----------
        parent : QtGui.QWidget
            The widget / dialog that will host the canvas layout
        """
        raise NotImplementedError('Include _init_canvas in your class declaration')

    def _init_controls(self, parent):
        """

        Parameters
        ----------
        parent : QtGui.QWidget
            The widget / dialog that will host the control layout

        """
        raise NotImplementedError('Include _init_controls in your class declaration')

    def _init_header(self, parent):
        self.lb_status = QtWidgets.QLabel('Label Status')
        self.lb_status.setStyleSheet("color:black")
        self.lb_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_warning = QtWidgets.QLabel()
        self.lb_warning.setStyleSheet('color:red')
        self.lb_warning.setAlignment(QtCore.Qt.AlignCenter)
        message_label = getattr(self.params, 'message_warn', '')
        self.Label = QtWidgets.QLabel(message_label)
        self.Label.setAlignment(QtCore.Qt.AlignLeft)

        parent.addWidget(self.lb_status)
        parent.addWidget(self.lb_warning)
        parent.addWidget(self.Label)
        parent.addStretch()
        message = getattr(self.params, 'init_message', '')
        self.update_status(message)

    def _init_footer(self, parent):
        """

        Parameters
        ----------
        parent : QtGui.QWidget
            The widget / dialog that will host the footer layout

        Returns
        -------
            The footer layout created
        """
        ctrl_layout = QtWidgets.QHBoxLayout()

        if sys.platform.startswith('darwin'):
            cmd_key = 'Cmd'
        else:
            cmd_key = 'Ctrl'

        self.btn_ok = QtWidgets.QPushButton(f'Save and Quit [{cmd_key}+S]')
        self.btn_undo = QtWidgets.QPushButton(f'Undo [{cmd_key}+Z]')

        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_undo)
        ctrl_layout.addWidget(self.btn_ok)

        self.btn_undo.clicked.connect(self.on_undo)
        self.btn_ok.clicked.connect(self.on_save_quit)

        parent.addLayout(ctrl_layout)
        return ctrl_layout

    def on_save_quit(self):
        self._controller.save()
        self.close()

    def on_undo(self):
        try:
            self._controller.undo()
        except InvalidActionWarning as err:
            self.update_warning(str(err))

    def show(self):
        """Override the base class show to fix a bug found in MAC"""
        super(BaseDialog, self).show()
        self.activateWindow()
        self.raise_()

    def update_status(self, msg):
        """Print the message into the dialog's status widget and clear the warning widget

        Parameters
        ----------
        msg : str  The message to display in the header of dialog
        """
        self.lb_status.setText(msg)
        self.lb_warning.setText('')

    def update_warning(self, msg):
        """Print the message into the dialog's warning widget and clear the status widget

        Parameters
        ----------
        msg : str  The message to display in the header of dialog
        """
        self.lb_warning.setText(msg)
        self.lb_status.setText('')


class BaseController(object):
    orientation = None
    _overlay_image = None
    _dialog = None
    default_position = ()
    position = ()
    saved = False

    def __init__(self, image, params, init_values=None):
        self.image = image
        self.params = params
        self.points = []
        self._overlay_image = init_values
        self.setup_intensity()

    def setup_intensity(self):
        if self.params.vmode == 'percentile':
            self.params.vmin, self.params.vmax = np.percentile(self.image.data,
                                                               (self.params.perc_min, self.params.perc_max))
        elif self.params.vmode == 'mean-std':
            # TODO: update this
            self.mean_intensity = (self.params.vmax + self.params.vmin) / 2.0
            self.std_intensity = (self.params.vmax - self.params.vmin) / 2.0
        elif self.params.vmode == 'clahe':
            # TODO: implement
            logger.warning("CLAHE is not implemented yet.")

    def reformat_image(self):
        """Set the camera position and increase contrast.

        The image orientation is set to SAL. And set the default contrast, and
        axes position for all canvases. Need to run before displaying the GUI
        with the image.

        """
        logger.debug(f'Image orientation {self.image.orientation}')
        self.orientation = self.image.orientation
        self.image.change_orientation('SAL')

        if self._overlay_image:
            self._overlay_image.change_orientation('SAL')

        x, y, z, t, dx, dy, dz, dt = self.image.dim
        self.params.aspect = dx / dy
        self.params.offset = x * dx
        self.default_position = Position(x // 2, y // 2, z // 2)

        self.setup_intensity()
        self.reset_position()

    def reset_position(self):
        """Set the canvas position to the center of the image"""
        self.position = self.default_position

    def valid_point(self, x, y, z):
        dim = self.image.dim
        if -1 < x < dim[0] and -1 < y < dim[1] and -1 < z < dim[2]:
            return True
        return False

    def save(self):
        logger.debug(f'Overlay shape {self._overlay_image.data.shape}')

        for point in self.points:
            x, y, z, label = [int(i) for i in point]
            self._overlay_image.data[x, y, z] = label

        if self.orientation != self._overlay_image.orientation:
            self._overlay_image.change_orientation(self.orientation)

        self.saved = True

    def undo(self):
        """Remove the last point selected and refresh the UI"""
        if self.points:
            x, y, z, label = self.points[-1]
            self.position = Position(x, y, z)
            self.points = self.points[:-1]
            self.label = label
            logger.debug(f'Point removed {self.position}')
        else:
            raise InvalidActionWarning('There is no points selected to undo')

    def as_string(self):
        if self._overlay_image is None:
            logger.warning('There is no information to save')
            return ''
        output = []
        data = self._overlay_image.data
        xs, ys, zs = np.where(data)
        for x, y, z in zip(xs, ys, zs):
            output.append(f'{x},{y},{z},{int(data[x, y, z])}')
        return ':'.join(output)

    def as_niftii(self, file_name=None):
        if not self._overlay_image:
            logger.warning('There is no information to save')
            raise IOError('There is no information to save')
        if file_name:
            self._overlay_image.absolutepath = file_name

        if self._overlay_image.absolutepath == self.image.absolutepath:
            raise IOError('Aborting: the original file and the labeled file are the same', self._overlay_image.absolutepath)

        logger.debug(f'Data: {np.where(self._overlay_image.data)}')
        self._overlay_image.save()


class TooManyPointsWarning(StopIteration):
    message = 'Reached the maximum number of points'


class InvalidActionWarning(ValueError):
    pass


class MissingLabelWarning(ValueError):
    pass


def launch_dialog(controller, dialog_class):
    app = QtWidgets.QApplication([])
    dialog = dialog_class(controller)
    dialog.show()
    app.exec_()
    return controller
