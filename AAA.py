# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'AAA.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5.QtCore import QDir, QModelIndex, QProcess
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeView, QFileSystemModel
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt  # 导入Qt模块Qt
import re  # 导入正则表达式模块


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1043, 677)
        self.selected_image_path = ""  # 用于保存选中的图片路径
        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(50, 130, 541, 371))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)

        # 检测按钮
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(770, 430, 141, 61))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.on_detect_button_clicked)  # 连接按钮点击信号到槽函数

        # 退出按钮
        self.Exit_Button = QtWidgets.QPushButton(Form)
        self.Exit_Button.setGeometry(QtCore.QRect(770, 540, 141, 61))
        self.Exit_Button.setObjectName("Exit_Button")
        self.Exit_Button.clicked.connect(self.on_exit_button_clicked)

        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.rootPath())

        self.treeView = QtWidgets.QTreeView(Form)
        self.treeView.setGeometry(QtCore.QRect(700, 130, 256, 192))
        self.treeView.setObjectName("treeView")
        self.treeView.setModel(self.model)
        self.treeView.setRootIndex(self.model.index("G:/yolo-train/yolov5-master/data/test_data"))  # 设置显示目录
        self.treeView.doubleClicked.connect(self.on_treeView_doubleClicked)  # 连接双击信号到槽函数

        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(340, 40, 341, 61))
        self.label.setObjectName("label")

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
        self.label.setText(_translate("Form",
                                      "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt;\">车牌检测系统</span></p></body></html>"))

    def translate_text(self, text):
        return (f"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                f"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                f"p, li {{ white-space: pre-wrap; }}\n"
                f"</style></head><body style=\" font-family:'SimSun'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                f"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin右; -qt-block-indent:0; text-indent:0px;\">{text}</p></body></html>")

    def on_treeView_doubleClicked(self, index: QModelIndex):
        # 获取双击的文件路径
        file_path = self.model.filePath(index)
        # 检查是否为图片文件
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            self.selected_image_path = file_path  # 更新选中的图片路径
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.scene.clear()  # 清空当前场景
                pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(pixmap_item)
                self.graphicsView.fitInView(pixmap_item, Qt.KeepAspectRatio)

    def on_detect_button_clicked(self):
        if self.selected_image_path:
            self.textBrowser.clear()  # 清空textBrowser内容
            self.textBrowser.setHtml(self.translate_text("<span style='font-size:16pt;'>正在检测中...</span>"))
            self.process = QProcess()
            self.process.setProgram("python")
            self.process.setArguments(["A_ccpd_detect_LPR_combine_5_15.py", "--source", self.selected_image_path])
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.start()

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")

        # 清空 "正在检测中..." 文本，然后添加检测结果
        if self.textBrowser.toPlainText().strip() == "正在检测中...":
            self.textBrowser.clear()

        # 使用正则表达式提取车牌号
        license_plate_pattern = re.compile(r'川[A-Z0-9]{6}')  # 假设车牌号格式为“川”开头的6个字符
        matches = license_plate_pattern.findall(stdout)

        # 在textBrowser上显示车牌号
        if matches:
            for match in matches:
                self.textBrowser.append(match)
        else:
            self.textBrowser.append(stdout.strip())  # 如果没有匹配到车牌号，显示原始输出

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8")
        self.textBrowser.append(f"<span style='color:red;'>{stderr}</span>")

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