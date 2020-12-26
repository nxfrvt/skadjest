import sys

from PyQt5.QtWidgets import QApplication

from window import Window


class App(QApplication):

    def __init__(self):
        super().__init__(sys.argv)
        self.window = Window()
        sys.exit(self.exec())


if __name__ == '__main__':
    app = App()
