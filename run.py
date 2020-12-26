import json
import pathlib
import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog

from window import Window


def load_photo(window: Window) -> None:
    """Prompts a file dialog and then
    loads chosen image file"""
    window.statusBar().showMessage("File explorer opened")

    path, _filter = QFileDialog.getOpenFileName(
        parent=window,
        caption=window.tr("Open file"),
        directory='c:\\users\\mazur\\desktop',
        filter=window.tr("Image files (*.png *.jpg *.gif)"))
    window.photo.setPixmap(QPixmap(path))
    window.statusBar().showMessage("Image loaded")


def read_plate(window: Window) -> None:
    """Reads the lbl_plate from photo and
    updates the corresponding label"""
    window.statusBar().showMessage("Reading the plate from the image")
    window.lbl_plate.setText(f"Plate read from the image:\n WU6337A")
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


class App(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.window = Window()
        sys.exit(self.exec())


if __name__ == '__main__':
    app = App()
