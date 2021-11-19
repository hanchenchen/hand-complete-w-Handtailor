from PyQt5.QtWidgets import QMainWindow
from PyQt5.Qt import QMessageBox
from . import users, signin
from collections import OrderedDict
import sqlite3
import numpy as np
import io
import os
import json


db_path = os.path.expanduser('~')
sqlite_data = os.path.join(db_path, 'Users.db')

# Users UI
class SignIn(QMainWindow, signin.Ui_Form):

    def __init__(self, parent=None):
        super(SignIn, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("登录")

        # UI connect
        self.confirm_button.clicked.connect(self.on_Confirm_clicked)
        self.cancel_button.clicked.connect(self.on_Cancel_clicked)
        
        self.init_params = None

    def on_Confirm_clicked(self):
        self.current_user = self.username_edit.text()
        connect = sqlite3.connect(sqlite_data)
        cursor = connect.cursor()
        sql = "SELECT * FROM data WHERE username=?"
        result = cursor.execute(sql, (self.current_user,))  # 执行sqlite语句
        connect.commit()
        data = result.fetchall()[0]  # 获取所有的内容
        self.sickside = data[3]
        self.user_shape = convert_array(data[-2]).reshape(2, -1)
        self.init_params = convert_array(data[-1]).reshape(2, -1)
        data_dict = data[4].replace("15", "\"15\"")
        data_dict = data_dict.replace("28", "\"28\"")
        print(data_dict)
        self.dict_elec_threshold = json.loads(data_dict)
        cursor.close()
        connect.close()
        self.close()

    def on_Cancel_clicked(self):
        self.username_edit.setText("")
        self.close()

    def update_initpose(self, opt_params):
        connect = sqlite3.connect(sqlite_data)
        cursor = connect.cursor()
        sql = 'UPDATE data SET initpose=? WHERE username=?'
        init_params = opt_params.ravel()
        cursor.execute(sql, (init_params, self.current_user))
        connect.commit()
        cursor.close()
        connect.close()


class SignUp(QMainWindow, users.Ui_Form):

    def __init__(self, parent=None):
        super(SignUp, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("注册")

        connect = sqlite3.connect(sqlite_data)
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        cursor = connect.cursor()
        sql = "CREATE TABLE IF NOT EXISTS data(username TEXT, age TEXT, gender TEXT, sickside TEXT, elec_threshold TEXT, handshape array, initpose array)"
        cursor.execute(sql)
        connect.commit()
        connect.close()
        self.handshape = np.zeros((2, 10))
        self.initpose = np.zeros((2, 62))

        # UI connect
        # self.confirm.clicked.connect(self.on_Confirm_clicked)
        # self.cancel.clicked.connect(self.on_Cancel_clicked)

    def on_Confirm_clicked(self):
        username = self.username.text()
        age = self.age.text()
        gender = self.gender.currentText()
        sickside = self.sickside.currentText()
        elec_threshold = str(self.user_elecThreshold_dict).replace("'", "\"")

        if not username or not age:
            QMessageBox.information(self, 'Error', "用户名或者年龄为空", QMessageBox.Yes)
        elif self.is_has(username):
            QMessageBox.information(self, 'Error', "该用户已存在", QMessageBox.Yes)
        else:
            connect = sqlite3.connect(sqlite_data, detect_types=sqlite3.PARSE_DECLTYPES)
            cursor = connect.cursor()
            sql = 'INSERT INTO data (username, age, gender, sickside, elec_threshold, handshape, initpose) VALUES(?, ?, ?, ?, ?, ?, ?)'  # 添加到数据库
            gender = "male" if gender == "男" else "female"
            sickside = "left" if sickside == "左" else "right"
            handshape = self.handshape  # (2, 10), 前10维是左手的形状，后10维是右手的形状
            initpose = self.initpose
            cursor.execute(sql, (username, age, gender, sickside, elec_threshold, handshape, initpose))
            connect.commit()
            cursor.close()
            connect.close()
            QMessageBox.information(self, "Successfully", "注册成功", QMessageBox.Yes)
            self.close()  # 注册完关闭窗口

    def on_Cancel_clicked(self):
        """
        清空输入框，下拉框恢复原样
        """
        self.username.setText("")
        self.age.setText("")
        self.gender.setCurrentIndex(0)
        self.sickside.setCurrentIndex(0)
        self.close()

    def closeEvent(self, event):
        """
        关闭后将输入框清空
        """
        self.on_Cancel_clicked()

    @staticmethod
    def is_has(username):
        """
        判断数据库中是否含有用户名
        """
        connect = sqlite3.connect(sqlite_data)
        cursor = connect.cursor()
        sql = 'SELECT * FROM data WHERE username=?'
        result = cursor.execute(sql, (username,))  # 执行sqlite语句
        connect.commit()
        data = result.fetchall()  # 获取所有的内容
        cursor.close()
        connect.close()
        if data:
            return True
        else:
            return False
    
    def update_handshape(self, opt_params):
        self.handshape = opt_params[:, 45:55]
        self.initpose = opt_params


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)
