# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'users.ui'
#
# Created by: PyQt5 UI code generator 5.12.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(413, 345)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(400, 300))
        self.layoutWidget = QtWidgets.QWidget(Form)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 40, 309, 74))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.username = QtWidgets.QLineEdit(self.layoutWidget)
        self.username.setObjectName("username")
        self.horizontalLayout.addWidget(self.username)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.age = QtWidgets.QLineEdit(self.layoutWidget)
        self.age.setObjectName("age")
        self.horizontalLayout_2.addWidget(self.age)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.gender = QtWidgets.QComboBox(self.layoutWidget)
        self.gender.setObjectName("gender")
        self.gender.addItem("")
        self.gender.addItem("")
        self.horizontalLayout_3.addWidget(self.gender)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.sickside = QtWidgets.QComboBox(self.layoutWidget)
        self.sickside.setObjectName("sickside")
        self.sickside.addItem("")
        self.sickside.addItem("")
        self.horizontalLayout_4.addWidget(self.sickside)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.layoutWidget1 = QtWidgets.QWidget(Form)
        self.layoutWidget1.setGeometry(QtCore.QRect(30, 120, 341, 201))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.layoutWidget1)
        self.groupBox.setObjectName("groupBox")
        self.get_electricity_threshold_start = QtWidgets.QPushButton(self.groupBox)
        self.get_electricity_threshold_start.setGeometry(QtCore.QRect(0, 30, 104, 33))
        self.get_electricity_threshold_start.setMinimumSize(QtCore.QSize(104, 33))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.get_electricity_threshold_start.setFont(font)
        self.get_electricity_threshold_start.setObjectName("get_electricity_threshold_start")
        self.get_electricity_threshold_capture = QtWidgets.QPushButton(self.groupBox)
        self.get_electricity_threshold_capture.setGeometry(QtCore.QRect(110, 30, 104, 33))
        self.get_electricity_threshold_capture.setMinimumSize(QtCore.QSize(104, 33))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.get_electricity_threshold_capture.setFont(font)
        self.get_electricity_threshold_capture.setObjectName("get_electricity_threshold_capture")
        self.switchGesture = QtWidgets.QPushButton(self.groupBox)
        self.switchGesture.setGeometry(QtCore.QRect(220, 30, 104, 33))
        self.switchGesture.setMinimumSize(QtCore.QSize(104, 33))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.switchGesture.setFont(font)
        self.switchGesture.setObjectName("switchGesture")
        self.verticalLayout_3.addWidget(self.groupBox)
        self.get_handshape = QtWidgets.QPushButton(self.layoutWidget1)
        self.get_handshape.setMinimumSize(QtCore.QSize(104, 33))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.get_handshape.setFont(font)
        self.get_handshape.setObjectName("get_handshape")
        self.verticalLayout_3.addWidget(self.get_handshape)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(-1, -1, -1, 10)
        self.horizontalLayout_6.setSpacing(20)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.cancel = QtWidgets.QPushButton(self.layoutWidget1)
        self.cancel.setMinimumSize(QtCore.QSize(97, 33))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.cancel.setFont(font)
        self.cancel.setObjectName("cancel")
        self.horizontalLayout_6.addWidget(self.cancel)
        self.confirm = QtWidgets.QPushButton(self.layoutWidget1)
        self.confirm.setMinimumSize(QtCore.QSize(96, 33))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.confirm.setFont(font)
        self.confirm.setObjectName("confirm")
        self.horizontalLayout_6.addWidget(self.confirm)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "用户名"))
        self.label_2.setText(_translate("Form", "年   龄"))
        self.label_3.setText(_translate("Form", "性   别"))
        self.gender.setItemText(0, _translate("Form", "男"))
        self.gender.setItemText(1, _translate("Form", "女"))
        self.label_4.setText(_translate("Form", "患   侧"))
        self.sickside.setItemText(0, _translate("Form", "左"))
        self.sickside.setItemText(1, _translate("Form", "右"))
        self.groupBox.setTitle(_translate("Form", "记录电流阈值"))
        self.get_electricity_threshold_start.setText(_translate("Form", "开始"))
        self.get_electricity_threshold_capture.setText(_translate("Form", "记录"))
        self.switchGesture.setText(_translate("Form", "切换手势"))
        self.get_handshape.setText(_translate("Form", "提取手部形状"))
        self.cancel.setText(_translate("Form", "取消"))
        self.confirm.setText(_translate("Form", "确认"))


