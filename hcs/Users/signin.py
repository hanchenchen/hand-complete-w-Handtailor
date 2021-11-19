# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Users/signin.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 300)
        self.widget = QtWidgets.QWidget(Form)
        self.widget.setGeometry(QtCore.QRect(90, 90, 224, 43))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 12)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.username_edit = QtWidgets.QLineEdit(self.widget)
        self.username_edit.setMinimumSize(QtCore.QSize(0, 27))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.username_edit.setFont(font)
        self.username_edit.setObjectName("username_edit")
        self.horizontalLayout.addWidget(self.username_edit)
        self.widget1 = QtWidgets.QWidget(Form)
        self.widget1.setGeometry(QtCore.QRect(100, 180, 174, 44))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 10)
        self.horizontalLayout_2.setSpacing(12)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.cancel_button = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cancel_button.setFont(font)
        self.cancel_button.setObjectName("cancel_button")
        self.horizontalLayout_2.addWidget(self.cancel_button)
        self.confirm_button = QtWidgets.QPushButton(self.widget1)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.confirm_button.setFont(font)
        self.confirm_button.setObjectName("confirm_button")
        self.horizontalLayout_2.addWidget(self.confirm_button)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "用户名"))
        self.cancel_button.setText(_translate("Form", "取消"))
        self.confirm_button.setText(_translate("Form", "确认"))

