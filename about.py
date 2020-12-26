from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout


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
