import sys

from PyQt5.QtWidgets import QApplication

from gui import Window


class App(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.window = Window()
        self.setStyle('Fusion')
        sys.exit(self.exec())


if __name__ == '__main__':
    app = App()

