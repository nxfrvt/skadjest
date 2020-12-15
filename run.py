import json
import pathlib
import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog

from window import Window


def train_network() -> None:  # TODO
    """Trains the network with provided data"""


class App(QApplication):

    def __init__(self):
        super().__init__(sys.argv)

        self.plate = "WU6337A"  # TODO
        self.window = Window()
        self.init_events()

        sys.exit(self.exec())

    def load_photo(self) -> None:
        """Prompts a file dialog and then
        loads chosen image file"""
        path, _filter = QFileDialog.getOpenFileName(
            parent=self.window,
            caption=self.window.tr("Open file"),
            directory='c:\\users\\mazur\\desktop',
            filter=self.window.tr("Image files (*.png *.jpg *.gif)"))
        self.window.photo.setPixmap(QPixmap(path))

    def read_plate(self) -> None:
        """Reads the lbl_plate from photo and
        updates the corresponding label"""
        self.window.lbl_plate.setText(f"Plate read from the image:\n{self.plate}")
        self.window.lbl_province.setText("Matching province couldn't be found")
        self.detect_province()

    def detect_province(self) -> None:
        province_char = self.plate[0]

        path = str(pathlib.Path(__file__).parent.absolute())
        with open(path + '/provinces.json') as json_file:
            data = json.load(json_file)
            for p in data['provinces']:
                if province_char == p['symbol']:
                    province = p['province']
                    self.window.lbl_province.setText(f"Found province: {province}")

    def init_events(self) -> None:
        self.window.btn_train.clicked.connect(lambda: print("training..."))  # TODO podpiecie pod trenowanie
        self.window.btn_open.clicked.connect(lambda: self.load_photo())
        self.window.btn_read.clicked.connect(lambda: self.read_plate())


if __name__ == '__main__':
    app = App()
