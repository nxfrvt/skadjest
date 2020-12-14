import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, \
    QGridLayout, QPushButton, QLabel, QFileDialog


class Window(QWidget):

    def __init__(self):
        """Ctor"""
        super().__init__()
        self.photo = QLabel("Your photo will appear here once loaded")
        self.photo.setScaledContents(True)
        self.init_ui()

    def init_ui(self):
        """User interface initialization"""
        layout = QGridLayout()

        btn_train = QPushButton("Train")
        btn_train.clicked.connect(lambda: print("Training..."))
        btn_open = QPushButton("Open a photo")
        btn_open.clicked.connect(lambda: self.load_photo())
        btn_read = QPushButton("Read the plate")
        layout.addWidget(btn_train, 0, 0)
        layout.addWidget(btn_open, 3, 0)
        layout.addWidget(btn_read, 5, 0)

        layout.addWidget(self.photo, 0, 1, 10, 3)
        layout.addWidget(QLabel("Plate read from the image: "), 7, 0)
        layout.addWidget(QLabel("Found state: \n"), 8, 0)

        self.setLayout(layout)
        self.setMinimumSize(800, 500)
        self.setWindowTitle('SkadJest')
        self.center()
        self.show()

    def center(self):
        """Centers the window on the screen"""
        qt_rectangle = self.frameGeometry()
        center_point = QDesktopWidget().availableGeometry().center()
        qt_rectangle.moveCenter(center_point)
        self.move(qt_rectangle.topLeft())

    def load_photo(self) -> None:
        """Prompts a file dialog and then
        loads chosen image file"""
        path, _filter = QFileDialog.getOpenFileName(
            parent=self,
            caption=self.tr("Open file"),
            directory='c:\\users\\mazur\\desktop',
            filter=self.tr("Image files (*.jpg *.gif)"))
        self.photo.setPixmap(QPixmap(path))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
