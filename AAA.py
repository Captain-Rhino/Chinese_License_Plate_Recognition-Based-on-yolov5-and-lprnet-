from PyQt5.QtCore import QDir, QModelIndex, QProcess
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QFileSystemModel, QPushButton
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import re
import os

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1043, 677)
        self.selected_image_path = ""
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(50, 130, 541, 371))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        #检测按钮
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(770, 430, 141, 61))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.on_detect_button_clicked)

        #退出按钮
        self.Exit_Button = QtWidgets.QPushButton(Form)
        self.Exit_Button.setGeometry(QtCore.QRect(770, 540, 141, 61))
        self.Exit_Button.setObjectName("Exit_Button")
        self.Exit_Button.clicked.connect(self.on_exit_button_clicked)

        #返回目录-按钮
        self.Back_Button = QPushButton(Form)
        self.Back_Button.setGeometry(QtCore.QRect(700, 80, 256, 40))
        self.Back_Button.setObjectName("Back_Button")
        self.Back_Button.setText("返回上一层")
        self.Back_Button.clicked.connect(self.on_back_button_clicked)


        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())

        #文件夹树~
        self.treeView = QtWidgets.QTreeView(Form)
        self.treeView.setGeometry(QtCore.QRect(700, 130, 256, 192))
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index("G:/yolo-train/yolov5-master/pic"))
        self.treeView.doubleClicked.connect(self.on_treeView_doubleClicked)


        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(340, 40, 341, 61))
        self.label.setObjectName("label")

        #打印输出用
        self.textBrowser = QtWidgets.QTextBrowser(Form)
        self.textBrowser.setGeometry(QtCore.QRect(50, 520, 531, 81))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setHtml(self.translate_text("<span style='font-size:16pt;'>检测到车牌：</span>"))

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "车牌检测系统"))
        self.pushButton.setText(_translate("Form", "开始检测"))
        self.Exit_Button.setText(_translate("Form", "结束进程"))
        self.label.setText(_translate("Form", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt;\">车牌检测系统</span></p></body></html>"))

    def translate_text(self, text):
        return (f"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                f"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                f"p, li {{ white-space: pre-wrap; }}\n"
                f"</style></head><body style=\" font-family:'SimSun'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">{text}</p></body></html>")

    def on_treeView_doubleClicked(self, index: QModelIndex):
        file_path = self.model.filePath(index)
        if self.model.isDir(index):
            self.treeView.setRootIndex(self.model.index(file_path))
        else:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.selected_image_path = file_path
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    self.scene.clear()
                    pixmap_item = QGraphicsPixmapItem(pixmap)
                    self.scene.addItem(pixmap_item)
                    self.graphicsView.fitInView(pixmap_item, Qt.KeepAspectRatio)

    def on_back_button_clicked(self):
        current_index = self.treeView.rootIndex()
        if current_index.isValid():
            parent_index = current_index.parent()
            if parent_index.isValid():
                self.treeView.setRootIndex(parent_index)

    def on_detect_button_clicked(self):
        if self.selected_image_path:
            self.textBrowser.clear()
            self.textBrowser.setHtml(self.translate_text("<span style='font-size:16pt;'>正在检测中...</span>"))
            self.process = QProcess()
            self.process.setProgram("python")
            self.process.setArguments(["A_ccpd_detect_LPR_combine_5_15.py", "--source", self.selected_image_path])
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.finished.connect(self.on_detection_finished)
            self.process.start()

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        self.textBrowser.append(stdout)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.textBrowser.append(f"<span style='color:red;'>{stderr}</span>")

    def on_detection_finished(self):
        text_content = self.textBrowser.toPlainText()
        # print("Text content:", text_content)  # browser输出
        match = re.search(r"runs[\\/]detect[\\/](exp\d+)", text_content, re.IGNORECASE)
        if match:
            dir_path = f"G:/yolo-train/yolov5-master/runs/detect/{match.group(1)}"
            # print("Extracted directory path:", dir_path)  # 文件夹地址
            if os.path.isdir(dir_path):
                image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                if image_files:
                    image_path = os.path.join(dir_path, image_files[0])
                    # print("Image file found:", image_path)  # 找到的图片地址
                    pixmap = QPixmap(image_path)
                    if not pixmap.isNull():
                        self.scene.clear()
                        pixmap_item = QGraphicsPixmapItem(pixmap)
                        self.scene.addItem(pixmap_item)
                        self.graphicsView.fitInView(pixmap_item, Qt.KeepAspectRatio)
                    # else:
                        # print("Pixmap is null. Could not load image.")  # 如果打不开图片
                # else:
                    # print("No image files found in the directory.")  # 如果地址里面没有图片
            # else:
                # print("Directory does not exist.")  # 如果地址不存在
        # else:
            # print("No match found for the directory path.")  # 如果没有匹配地址

    def on_exit_button_clicked(self):
        QApplication.quit()

if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow
    app = QApplication(sys.argv)
    Form = QMainWindow()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
