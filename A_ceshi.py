import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QFileSystemModel
from PyQt5.QtCore import QDir

class MyForm(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('File Tree View')

        self.treeView = QTreeView(self)
        self.treeView.setGeometry(50, 50, 700, 500)

        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())  # 设置根路径为系统根目录
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index("G:/yolo-train/yolov5-master/data/test_data"))  # 设置显示目录

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyForm()
    window.show()
    sys.exit(app.exec_())
