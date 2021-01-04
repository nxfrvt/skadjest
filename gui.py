import json
import pathlib
import sys

from PyQt5 import QtCore
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QIcon, QPixmap, QRegExpValidator
from PyQt5.QtWidgets import QWidget, QDesktopWidget, \
    QGridLayout, QPushButton, QLabel, QMainWindow, QAction, QMenuBar, QFileDialog, QVBoxLayout, QDialog, QPlainTextEdit, \
    QTextEdit, QLineEdit, QSizePolicy

from extract import find_license_plate, LicensePlateException
from neural_network import NeuralNetwork as nn

nn = nn()


class AboutDialog(QDialog):

    def __init__(self, *args, **kwargs):
        super(AboutDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("About")
        self.setWindowIcon(QIcon('rsc/icons/about.png'))
        self.setFixedSize(300, 100)

        self.label = QLabel("Authors\n\nBartłomiej Mazurek\nMichał Staruch")
        self.label.setAlignment(QtCore.Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)


class Window(QMainWindow):

    def __init__(self):
        """Ctor"""
        super().__init__()
        self.menubar = QMenuBar()
        self.setMinimumSize(800, 500)
        self.setWindowTitle('SkadJest')
        self.setWindowIcon(QIcon('rsc/icons/app.png'))
        self.__center()
        self.show()

        self.plate = "None"
        self.path = ""
        self.photo = QLabel("Your photo will appear here once loaded")
        self.photo.setScaledContents(True)
        self.photo.setAlignment(QtCore.Qt.AlignCenter)

        self.btn_train = QPushButton("Train")
        self.btn_open = QPushButton("Open a photo")
        self.btn_read = QPushButton("Read the plate")

        self.lbl_plate = QLabel("Plate read from the image:")
        self.lbl_province = QLabel("Found province: ")
        self.txt_iterations = QLineEdit("10000")
        self.txt_iterations.setValidator(QRegExpValidator(QRegExp("[1-9][0-9]{1,4}")))  # allows training in 1-99999 range
        self.txt_iterations.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # no idea what it does
        # but w/o LineEdit is huge

        self.__init_ui()

    def __init_ui(self) -> None:
        """User interface initialization"""
        layout = QGridLayout()
        self.statusBar().showMessage("Starting the application.")

        #  add the menu bar at the top
        layout.addWidget(self.__init_menubar(), 0, 0, 1, 8)

        layout.addWidget(self.btn_train, 1, 0, 1, 2)
        layout.addWidget(self.txt_iterations, 2, 1, 1, 1)
        layout.addWidget(self.btn_open, 6, 0, 1, 2)
        layout.addWidget(self.btn_read, 8, 0, 1, 2)

        layout.addWidget(self.photo, 1, 2, 15, 6)
        layout.addWidget(self.lbl_plate, 10, 0, 1, 2)
        layout.addWidget(self.lbl_province, 12, 0, 1, 2)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        self.__init_events()
        self.statusBar().showMessage("Application ready")

    def __init_menubar(self) -> QMenuBar:
        #  File menu
        menu_file = self.menubar.addMenu('File')

        act_open = QAction(QIcon('rsc/icons/open.png'), '&Open', self)
        act_open.setShortcut('Ctrl+O')
        act_open.setStatusTip('Open image')
        act_open.triggered.connect(lambda: load_photo(self))
        menu_file.addAction(act_open)
        menu_file.addSeparator()

        act_read = QAction(QIcon('rsc/icons/read.png'), '&Read', self)
        act_read.setShortcut('Ctrl+R')
        act_read.setStatusTip('Read plate')
        act_read.triggered.connect(lambda: read_plate(self))
        menu_file.addAction(act_read)
        menu_file.addSeparator()

        act_exit = QAction(QIcon('rsc/icons/exit.png'), '&Exit', self)
        act_exit.setShortcut('Ctrl+Q')
        act_exit.setStatusTip('Exit application')
        act_exit.triggered.connect(lambda: sys.exit())
        menu_file.addAction(act_exit)

        #  Help menu
        menu_help = self.menubar.addMenu("Help")

        act_about = QAction(QIcon('rsc/icons/about.png'), '&About', self)
        act_about.setShortcut('Ctrl+A')
        act_about.setStatusTip('About')
        act_about.triggered.connect(lambda: self.__about())
        menu_help.addAction(act_about)

        return self.menubar

    def __init_events(self) -> None:
        """Connects all events to corresponding widgets"""
        self.btn_train.clicked.connect(lambda: train(self))
        self.btn_open.clicked.connect(lambda: load_photo(self))
        self.btn_read.clicked.connect(lambda: read_plate(self))

    def __center(self) -> None:
        """Centers the window on the screen"""
        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())

    def __about(self) -> None:
        """Displays the About dialog"""
        dlg = AboutDialog(self)
        dlg.exec_()


def train(window: Window):
    window.txt_iterations.setVisible(False)
    iterations = int(window.txt_iterations.text())
    if iterations == "":
        pass
        # TODO wymus wprowadzenia numeru(dialog?)
    else:
        nn.train(iterations)
        window.btn_train.setEnabled(False)
        window.statusBar().showMessage("Network trained")


def load_photo(window: Window):
    """Prompts a file dialog and then
    loads chosen image file"""
    window.statusBar().showMessage("File explorer opened")

    path, _filter = QFileDialog.getOpenFileName(
        parent=window,
        caption=window.tr("Open file"),
        directory='rsc/photos',
        filter=window.tr("Image files (*.png *.jpg *.gif)"))
    window.photo.setPixmap(QPixmap(path))
    window.path = path
    window.statusBar().showMessage("Image loaded")


def read_plate(window: Window) -> None:
    """Reads the lbl_plate from photo and
    updates the corresponding label"""
    window.statusBar().showMessage("Reading the plate from the image")
    try:
        find_license_plate(window.path)
    except LicensePlateException:
        print("damn error again")
        # TODO dialog
    plate = ""
    #TODO wykrycie ilosc jpgow w tmp i ustawienie takie range
    for photo in range(1, 8):
        plate += nn.estimate("tmp/" + str(photo) + ".jpg")
    window.lbl_plate.setText(f"Plate read from the image:\n {plate}")
    window.statusBar().showMessage("Looking for matching province symbols")
    window.lbl_province.setText("Matching province couldn't be found")
    detect_province(window)
    window.statusBar().showMessage("Application ready")


def detect_province(window: Window) -> None:
    """Detects the province from the plate read from the image"""
    province_char = window.lbl_plate.text()[28]
    path = str(pathlib.Path(__file__).parent.absolute())
    with open(path + '/provinces.json') as json_file:
        data = json.load(json_file)
        for p in data['provinces']:
            if province_char == p['symbol']:
                province = p['province']
                window.lbl_province.setText(f"Found province: {province}")
