from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QDesktopWidget, \
    QGridLayout, QPushButton, QLabel, QMainWindow


class Window(QMainWindow):

    def __init__(self):
        """Ctor"""
        super().__init__()

        self.plate = "None"
        self.photo = QLabel("Your photo will appear here once loaded")
        self.photo.setScaledContents(True)
        self.photo.setAlignment(QtCore.Qt.AlignCenter)

        self.lbl_plate = QLabel("Plate read from the image:")
        self.lbl_province = QLabel("Found province: ")

        self.btn_train = QPushButton("Train")
        self.btn_open = QPushButton("Open a photo")
        self.btn_read = QPushButton("Read the plate")
        self.init_ui()

    def init_ui(self) -> None:
        """User interface initialization"""
        layout = QGridLayout()

        layout.addWidget(self.btn_train, 0, 0)
        layout.addWidget(self.btn_open, 3, 0)
        layout.addWidget(self.btn_read, 5, 0)

        layout.addWidget(self.photo, 0, 1, 10, 3)
        layout.addWidget(self.lbl_plate, 7, 0)
        layout.addWidget(self.lbl_province, 8, 0)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)
        self.setMinimumSize(800, 500)
        self.setWindowTitle('SkadJest')
        self.center()
        self.show()

    def center(self) -> None:
        """Centers the window on the screen"""
        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())
