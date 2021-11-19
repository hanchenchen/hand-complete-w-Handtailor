#！、usr/bin/env.python
#._*_ coding:utf-8 _*_

from PyQt5.QtCore import Qt,QPoint,QFile,QTextStream
from PyQt5.QtWidgets import QHBoxLayout,QLabel,QPushButton,QWidget, QMessageBox
import logging, psutil
import os

#用于设置按钮的属性
def load_windowstyle():
    f = QFile("BorderlessWindow.css")
    if not f.exists():
        print("Unable to load stylesheet, file not found in resources")
        return ""
    else:
        f.open(QFile.ReadOnly|QFile.Text)
        ts = QTextStream(f)
        stylesheet = ts.readAll()
        return stylesheet


class MyBar(QWidget):

    def __init__(self,parent):
        super(MyBar, self).__init__()
        self.parent = parent
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 5, 10, 0)
        self.title = QLabel("康复手势对侧训练监控系统")

        self.m_DragPosition = self.parent.pos()

        self.pushButtonClose = QPushButton("")
        self.pushButtonClose.setObjectName("pushButtonClose")
        self.pushButtonClose.clicked.connect(self.btn_close_clicked)

        self.pushButtonMinimize = QPushButton("")
        self.pushButtonMinimize.setObjectName("pushButtonMinimize")
        self.pushButtonMinimize.clicked.connect(self.btn_min_clicked)

        self.pushButtonMaximize = QPushButton("")
        self.pushButtonMaximize.setObjectName("pushButtonMaximize")
        self.pushButtonMaximize.clicked.connect(self.btn_max_clicked)

        self.title.setFixedHeight(14)
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.pushButtonMinimize)
        self.layout.addWidget(self.pushButtonMaximize)
        self.layout.addWidget(self.pushButtonClose)

        self.windowstyle = load_windowstyle()
        self.setStyleSheet(self.windowstyle)
        self.setLayout(self.layout)

        self.pushButtonClose.setShortcut('Esc')
        self.start = QPoint(0, 0)
        self.pressing = False
        self.maxNormal = True

    def resizeEvent(self, QResizeEvent):
        super(MyBar, self).resizeEvent(QResizeEvent)
        self.title.setFixedWidth(self.parent.width())

    def mousePressEvent(self, event):
        self.start = self.mapToGlobal(event.pos())
        self.pressing = True

    def mouseMoveEvent(self, event):
        if self.pressing:
            self.end = self.mapToGlobal(event.pos())
            self.movement = self.end-self.start
            self.parent.setGeometry(self.mapToGlobal(self.movement).x(),
                                    self.mapToGlobal(self.movement).y(),
                                    self.parent.width(), self.parent.height())
            self.start = self.end

    def mouseReleaseEvent(self, event):
        self.pressing = False

    def btn_close_clicked(self):
        # PROCNAME = "python.exe"
        # for proc in psutil.process_iter():
        #     if proc.name() == PROCNAME:
        #         proc.kill()
        reply = QMessageBox.question(self,
                                     '康复手势対侧训练监控系统',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.parent.close()
            PROCNAME = "python.exe"
            for proc in psutil.process_iter():
                if proc.name() == PROCNAME:
                    proc.kill()
            # os._exit(0)

    def btn_max_clicked(self):
        if not self.maxNormal:
            self.parent.showNormal()
            self.maxNormal = True
            self.pushButtonMaximize.setObjectName("pushButtonMaximize")
            self.pushButtonMaximize.setStyleSheet(self.windowstyle)
            #self.pushButtonMaximize.setIcon(QIcon('Icons/Maximize.png'))
        else:
            self.parent.showMaximized()
            self.maxNormal = False
            self.pushButtonMaximize.setObjectName("pushButtonRestore")
            self.pushButtonMaximize.setStyleSheet(self.windowstyle)
            #self.pushButtonMaximize.setIcon(QIcon('Icons/Restore.png'))

    def btn_min_clicked(self):
        self.parent.showMinimized()






