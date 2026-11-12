"""
Qt widgets for manual labeling of images

Copyright (c) 2017 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""

import logging
from time import time

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

from PyQt5 import QtCore, QtGui, QtWidgets

from spinalcordtoolbox.gui.base import MissingLabelWarning

logger = logging.getLogger(__name__)


class VertebraeWidget(QtWidgets.QWidget):
    """A group of checkboxes that list labels."""
    _unchecked = []
    _checked = []
    _active_label = None
    _check_boxes = {}
    _labels = None
    _label = None

    def __init__(self, parent, vertebraes):
        super(VertebraeWidget, self).__init__(parent)
        self.parent = parent
        self.vertebraes = vertebraes
        self._init_ui(parent.params)
        self.refresh()

    def _init_ui(self, params):
        layout = QtWidgets.QVBoxLayout()
        self.setLayout(layout)
        font = QtGui.QFont()
        font.setPointSize(10)

        for vertebrae in self.vertebraes:
            rdo = QtWidgets.QCheckBox(f'Label {vertebrae}')
            rdo.label = vertebrae
            rdo.setFont(font)
            rdo.setTristate()
            self._check_boxes[vertebrae] = rdo
            rdo.clicked.connect(self.on_select_label)
            layout.addWidget(rdo)

        layout.addStretch()

    def on_select_label(self):
        label = self.sender()
        self.label = label.label

    def refresh(self, labels=None):
        if labels:
            self._checked = labels
            self._unchecked = set(self._check_boxes.keys()) - set(labels)

        for checkbox in self._check_boxes.values():
            checkbox.setCheckState(QtCore.Qt.Unchecked)

        logger.debug(f'refresh labels {self.parent._controller.points}')
        for point in self.parent._controller.points:
            self._check_boxes[point[3]].setCheckState(QtCore.Qt.Checked)

    @property
    def label(self):
        if self._active_label:
            return self._active_label.label
        raise MissingLabelWarning('No vertebrae was selected')

    @label.setter
    def label(self, index):
        self.refresh()
        self._active_label = self._check_boxes[index]
        self._active_label.setCheckState(QtCore.Qt.PartiallyChecked)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, values):
        self._labels = values
        for x in self._check_boxes.values():
            x.setCheckState(QtCore.Qt.Unchecked)

        for label in self._labels:
            self._check_boxes[label].setCheckState(QtCore.Qt.Checked)


class AnatomicalCanvas(FigureCanvas):
    """Base canvas for anatomical views

    Attributes
    ----------
    point_selected_signal : QtCore.Signal
        Create a event when user clicks on the canvas

    """
    point_selected_signal = QtCore.Signal(float, float, float)
    _horizontal_nav = None
    _vertical_nav = None
    _navigation_state = False
    annotations = []
    last_update = 0
    update_freq = 0.0667
    previous_point = (0, 0)

    def __init__(self, parent, width=8, height=8, dpi=100, crosshair=False, plot_points=False,
                 annotate=False, vertical_nav=False, horizontal_nav=False):
        self._parent = parent
        self._image = parent.image
        self._params = parent.params
        self._crosshair = crosshair
        self._plot_points = plot_points
        self._annotate_points = annotate
        self._vertical_nav = vertical_nav
        self._horizontal_nav = horizontal_nav
        self.position = None

        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]

        self._fig = Figure(figsize=(width, height), dpi=dpi)
        super(AnatomicalCanvas, self).__init__(self._fig)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.vmin_updated = self._params.vmin
        self.vmax_updated = self._params.vmax

    def _init_ui(self, data, aspect):
        self._fig.canvas.mpl_connect('button_release_event', self.on_select_point)
        self._fig.canvas.mpl_connect('scroll_event', self.on_zoom)
        self._fig.canvas.mpl_connect('button_release_event', self.on_change_intensity)
        self._fig.canvas.mpl_connect('motion_notify_event', self.on_change_intensity)

        self._axes = self._fig.add_axes([0, 0, 1, 0.9], frameon=True)
        self._axes.axis('off')
        self.view = self._axes.imshow(
            data,
            cmap=self._params.cmap,
            interpolation=self._params.interp,
            vmin=self._params.vmin,
            vmax=self._params.vmax,
            alpha=self._params.alpha)
        self._axes.set_aspect(aspect)

        if self._crosshair:
            self.cursor = Cursor(self._axes, useblit=True, color='r', linewidth=1)

        self.points = self._axes.plot([], [], '.r', markersize=7)[0]

    def title(self, message):
        self._fig.suptitle(message, fontsize=10)

    def annotate(self, x, y, label):
        self.annotations.append(self._axes.annotate(label, xy=(x, y), xytext=(-3, 3),
                                                    textcoords='offset points', ha='right',
                                                    va='bottom', color='r'))

    def clear(self):
        for i in self.annotations:
            i.remove()
        self.annotations = []
        self.points.set_xdata([])
        self.points.set_ydata([])

    def refresh(self):
        self.view.set_clim(vmin=self.vmin_updated, vmax=self.vmax_updated)
        # self.view.set_clim(self._parent._controller.vmin_updated,
        #                    self._parent._controller.vmax_updated)
        logger.debug(f"vmin_updated={self.vmin_updated}, vmax_updated={self.vmax_updated}")
        self.plot_position()
        self.plot_points()
        self.view.figure.canvas.draw()

    def plot_data(self, xdata, ydata, labels):
        self.points.set_xdata(xdata)
        self.points.set_ydata(ydata)

        if self._annotate_points:
            for x, y, label in zip(xdata, ydata, labels):
                self.annotate(x, y, label)

    def on_zoom(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 'up':
            scale_factor = 1.3
        else:
            scale_factor = 1 / 1.3

        x = event.xdata
        y = event.ydata

        x_lim = self._axes.get_xlim()
        y_lim = self._axes.get_ylim()

        left = (x - x_lim[0]) * scale_factor
        right = (x_lim[1] - x) * scale_factor
        top = (y - y_lim[0]) * scale_factor
        bottom = (y_lim[1] - y) * scale_factor

        if x + right - left >= self.x_max or y + bottom - top >= self.y_max:
            return

        self._axes.set_xlim(x - left, x + right)
        self._axes.set_ylim(y - top, y + bottom)
        self.view.figure.canvas.draw()

    def on_select_point(self, event):
        pass

    def on_change_intensity(self, event):
        if event.xdata is None or event.ydata is None:
            return

        if event.button == 3:  # right click
            curr_time = time()

            if curr_time - self.last_update <= self.update_freq:
                # TODO: never enters that loop because last_update set to 0 and it is never updated
                return

            if (abs(event.xdata - self.previous_point[0]) < 1 and abs(event.ydata - self.previous_point) < 1):
                # TODO: never enters that loop because previous_point set to 0,0 and it is never updated
                self.previous_point = (event.xdata, event.ydata)
                return

            logger.debug(f"X={event.xdata}, Y={event.ydata}")
            xlim, ylim = self._axes.get_xlim(), self._axes.get_ylim()
            x_factor = (event.xdata - xlim[0]) / float(xlim[1] - xlim[0])  # between 0 and 1. No change: 0.5
            y_factor = (event.ydata - ylim[1]) / float(ylim[0] - ylim[1])

            # get dynamic of the image
            vminvmax = self._params.vmax - self._params.vmin  # todo: get variable based on image quantization

            # adjust brightness by adding offset to image intensity
            # the "-" sign is there so that when moving the cursor to the right, brightness increases (more intuitive)
            # the 2.0 factor maximizes change.
            self.vmin_updated = self._params.vmin - (x_factor - 0.5) * vminvmax * 2.0
            self.vmax_updated = self._params.vmax - (x_factor - 0.5) * vminvmax * 2.0

            # adjust contrast by multiplying image dynamic by scaling factor
            # the factor 2.0 maximizes contrast change. For y_factor = 0.5, the scaling will be 1, which means no change
            # in contrast
            self.vmin_updated = self.vmin_updated * (y_factor * 2.0)
            self.vmax_updated = self.vmax_updated * (y_factor * 2.0)

            self.refresh()

    def horizontal_position(self, position):
        if self._horizontal_nav:
            try:
                self._horizontal_nav.remove()
            except AttributeError:
                pass
            self._horizontal_nav = self._axes.axhline(position, color='r')

    def vertical_position(self, position):
        if self._vertical_nav:
            try:
                self._vertical_nav.remove()
            except AttributeError:
                pass
            self._vertical_nav = self._axes.axvline(position, color='r')

    def __repr__(self):
        return f'{self.__class__}: {self._x}, {self._y}, {self._z}'

    def __str__(self):
        return f'{self._x}: {self._y}, {self._z}'


class SagittalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(SagittalCanvas, self).__init__(*args, **kwargs)
        x, y, z, _, dx, dy, dz, _ = self._image.dim
        self._init_ui(self._image.data[:, :, self._z], dx / dy)
        self.annotations = []
        self.x_max = y
        self.y_max = x

    def refresh(self):
        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]
        data = self._image.data[:, :, self._z]
        self.view.set_array(data)
        super(SagittalCanvas, self).refresh()

    def on_select_point(self, event):
        if event.xdata is not None and event.ydata is not None and event.button == 1:
            self.point_selected_signal.emit(event.ydata, event.xdata, self._z)

    def plot_points(self):
        """Plot the controller's list of points (x, y) and annotate the point with the label"""
        if self._plot_points:
            logger.debug(f'Plotting points {self._parent._controller.points}')
            points = self._parent._controller.points
            self.clear()
            try:
                xs, ys, zs, labels = zip(*points)
                self.plot_data(ys, xs, labels)
            except ValueError:
                pass

    def plot_position(self):
        position = self._parent._controller.position
        self.horizontal_position(position[0])
        self.vertical_position(position[1])


class CoronalCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(CoronalCanvas, self).__init__(*args, **kwargs)
        x, y, z, _, dx, dy, dz, _ = self._image.dim
        self._init_ui(self._image.data[:, self._y, :], dx / dz)
        self.x_max = x
        self.y_max = z

    def refresh(self):
        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]
        data = self._image.data[:, self._y, :]
        self.view.set_array(data)
        super(CoronalCanvas, self).refresh()

    def on_select_point(self, event):
        if event.xdata is not None and event.ydata is not None and event.button == 1:
            self.point_selected_signal.emit(event.xdata, self._y, event.ydata)

    def plot_points(self):
        logger.debug(f'Plotting points {self._parent._controller.points}')
        if self._parent._controller.points:
            points = [x for x in self._parent._controller.points]
            self.clear()
            try:
                xs, ys, zs, _ = zip(*points)
                self.plot_data(xs, zs, [])
            except ValueError:
                pass
            self.view.figure.canvas.draw()


class AxialCanvas(AnatomicalCanvas):
    def __init__(self, *args, **kwargs):
        super(AxialCanvas, self).__init__(*args, **kwargs)
        x, y, z, _, dx, dy, dz, _ = self._image.dim
        self._init_ui(self._image.data[self._x, :, :], dy / dz)
        self.x_max = z
        self.y_max = y

    def refresh(self):
        self._x, self._y, self._z = [int(i) for i in self._parent._controller.position]
        data = self._image.data[self._x, :, :]
        self.view.set_array(data)
        super(AxialCanvas, self).refresh()

    def on_select_point(self, event):
        if event.xdata is not None and event.ydata is not None and event.button == 1:
            self.point_selected_signal.emit(self._x, event.ydata, event.xdata)

    def plot_points(self):
        if self._plot_points:
            controller = self._parent._controller
            logger.debug(f'Plotting points {controller.points}')
            points = [x for x in controller.points if x[0] == controller.position[0]]
            self.clear()
            try:
                xs, ys, zs, _ = zip(*points)
                self.plot_data(zs, ys, [])
            except ValueError:
                pass

    def plot_position(self):
        position = self._parent._controller.position
        self.horizontal_position(position[1])
        self.vertical_position(position[2])


class AnatomicalToolbar(NavigationToolbar):
    def __init__(self, canvas, parent):
        self.toolitems = (('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
                          ('Back', 'Back to previous view', 'back', 'back'),
                          ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'))
        super(AnatomicalToolbar, self).__init__(canvas, parent)
