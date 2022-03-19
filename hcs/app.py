from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, QFile, QTextStream, QThreadPool, QSettings, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.Qt import QMessageBox
from Users import SignUp, SignIn, sqlite_data
from Settings import Preference
from backhend import Prepare_Worker, External_Worker, Estimate_TemporalSmoothing_Worker, Estimate_HandTailor_Worker
from collections import deque, OrderedDict
import multiprocessing as mp
from threading import Thread
import socket
import websocket
import chumpy
import torch
import numpy as np
import MyBar
import camera
import cv2
import sys
import time
import mesh_displayer
import os
import json
import psutil
import system
import sqlite3
import pandas.io.sql as sql

# from backhend.handtailor_solve import Solver as HandtailorSolver
# from backhend.video_temporal_lifter_solve import Solver as TemporalLifterSolver


def load_darkstyle():
    f = QFile("darkstyle/style.qss")
    if not f.exists():
        print("Unable to load stylesheet, file not in resources")
        return ""
    else:
        f.open(QFile.ReadOnly | QFile.Text)
        ts = QTextStream(f)
        stylesheet = ts.readAll()
        return stylesheet


class MainWindow(QWidget):
    delay_complete_signal = pyqtSignal()
    delay_prepare_signal = pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.layout = QVBoxLayout()
        w = QMainWindow()
        self.ui = system.Ui_MainWindow()
        self.ui.setupUi(w)
        self.registration_UI = SignUp(self)
        self.signin_UI = SignIn(self)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.title_bar = MyBar.MyBar(self)
        self.layout.addWidget(self.title_bar)
        self.layout.addWidget(w)
        self.count = 0

        # Config
        self.settings = QSettings("config.ini", QSettings.IniFormat)

        # Preference
        self.preference_UI = Preference(self.settings, self)

        # UI
        self.ui.action_on.triggered.connect(self.on_OpenDevice)
        self.ui.registration.triggered.connect(self.on_registration_triggered)
        self.ui.login.triggered.connect(self.on_login_triggered)
        self.ui.export_usersdata.triggered.connect(self.on_export_triggered)
        self.ui.action_preference.triggered.connect(self.on_preference_triggered)
        self.ui.action_singlecam_start.triggered.connect(self.on_singlecam_prepare_triggered)
        self.ui.action_singlecam_terminate.triggered.connect(self.on_singlecam_terminate_triggered)
        # self.ui.action_doublecam_start.triggered.connect(self.on_doublecam_prepare_triggered)
        # self.ui.action_doublecam_terminate.triggered.connect(self.on_doublecam_terminate_triggered)
        self.ui.action_fist.triggered.connect(self.on_action_fist_triggered)
        self.ui.action_wristforward.triggered.connect(self.on_action_wristforward_triggered)
        self.ui.action_wristbackward.triggered.connect(self.on_action_wristbackward_triggered)
        self.ui.action_wristside.triggered.connect(self.on_action_wristside_triggered)
        self.ui.StopEmergency.clicked.connect(self.on_StopEmergency_Clicked)
        self.preference_UI.apply_button.clicked.connect(self.on_preference_apply)
        self.preference_UI.confirm_button.clicked.connect(self.on_preference_confirm)
        self.registration_UI.get_handshape.clicked.connect(self.on_gethandshape_clicked)
        self.registration_UI.get_electricity_threshold_start.clicked.connect(self.on_getElectricityThresholdStart_clicked)
        self.registration_UI.get_electricity_threshold_capture.clicked.connect(self.on_getElectricityThresholdCapture_clicked)
        self.registration_UI.switchGesture.clicked.connect(self.on_switchGesture_clicked)
        self.registration_UI.confirm.clicked.connect(self.on_RegistrationUIConfirm_clicked)
        self.registration_UI.cancel.clicked.connect(self.on_RegistrationUICancel_clicked)
        for i in range(8):
            getattr(self.ui, "increase_channel_{}".format(i + 1)).clicked.connect(self.get_electricity_command_func("Increase", i))
            getattr(self.ui, "reduce_channel_{}".format(i + 1)).clicked.connect(self.get_electricity_command_func("Reduce", i))

        # Connect
        # self.camera.display_image_signal.connect(self.PutImg)
        self.delay_complete_signal.connect(self.on_gethandshape)
        self.delay_prepare_signal.connect(self.on_singlecam_prepare)
        self.title_bar.pushButtonClose.clicked.disconnect()
        self.title_bar.pushButtonClose.clicked.connect(self.quit_system)

        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setMinimumSize(866, 545)

        self.pressing = False
        self.maxNormal = True

        self.threadpool = QThreadPool()
        print("Multi-threading with maximum %d thread" % self.threadpool.maxThreadCount())

        self.status = 0
        self.num_start_triggered = 0
        self.gestures = ["fist", "wristforward", "wristbackward", "wristside"]
        self.evoke_socket()

    def PutImg(self, display_images):
        height = int(self.settings.value("CAMERA/HEIGHT"))
        width = int(self.settings.value("CAMERA/WIDTH"))
        if display_images.shape[0] == height:
            display_images = cv2.copyMakeBorder(display_images, height // 2, height // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        display_images = cv2.cvtColor(display_images, cv2.COLOR_BGR2RGB)
        qt_image = QImage(display_images.data,
                          2 * width,
                          2 * height,
                          display_images.strides[0],
                          QImage.Format_RGB888)
        pixmap = QPixmap(qt_image)
        pixmap = pixmap.scaled(self.ui.display.geometry().width(),
                               self.ui.display.geometry().height())
        self.ui.display.setPixmap(pixmap)

    def Display_Images(self):
        while True:
            if not self._device_output_queue.empty():
                meta = self._device_output_queue.get()
                if type(meta) == dict:
                    print('!! meta', meta)
                    self.PutImg(meta["display"])
                    self.inputs = meta
                else:
                    self._device_output_queue.put(meta)
                if self.status == 0:
                    break
                # if hasattr(self, "start_time"):
                #     print(time.clock() - self.start_time)

    def on_OpenDevice(self):
        # Using MultiProcessing
        self._device_input_queue = mp.Queue()
        self._device_output_queue = mp.Queue()
        width = int(self.settings.value("CAMERA/WIDTH"))
        height = int(self.settings.value("CAMERA/HEIGHT"))
        fps = int(self.settings.value("CAMERA/FPS"))
        self._worker_process1 = mp.Process(target=camera.Worker, args=(self._device_input_queue, self._device_output_queue,
                                                                       1, width, height, fps))
        self._worker_process1.start()

        self.status = 1
        display_thread = Thread(target=self.Display_Images, args=())
        display_thread.setDaemon(True)
        display_thread.start()
        self.ui.action_off.triggered.connect(self.on_ShutdownDevice)
        self.title_bar.pushButtonClose.clicked.disconnect()
        self.title_bar.pushButtonClose.clicked.connect(self.quit_system)
        # self.camera.start()
        # self.model_thread = Thread(target=self.load_models, args=())
        # self.model_thread.setDaemon(True)
        # self.model_thread.start()


    # def load_models(self):
    #     ks = self._device_output_queue.get()['ks'][0]
    #     self.TemporalLifterSolver = TemporalLifterSolver(ks)
    #     self.HandtailorSolver = HandtailorSolver(ks)

    def on_ShutdownDevice(self):
        self._device_input_queue.put('STOP')

    def on_registration_triggered(self):
        self.registration_UI.user_elecThreshold_dict = dict()
        self.registration_UI.show()

    def on_login_triggered(self):
        self.signin_UI.show()

    def on_preference_triggered(self):
        self.preference_UI.show()

    def start_mesh_display_process(self, R=None):
        # Multiprocessing display open3d
        self._vertices_queue = mp.Queue()
        self._output_queue = mp.Queue()
        window_size = max(int(self.settings.value("CAMERA/HEIGHT")), int(self.settings.value("CAMERA/WIDTH")))
        ks = self._device_output_queue.get()['ks'][0]
        self._worker_process4 = mp.Process(target=mesh_displayer.Worker, args=(self._vertices_queue, self._output_queue,
                                                                               7, window_size, ks, R))
        self._worker_process4.start()
        self.title_bar.pushButtonClose.clicked.disconnect()
        self.title_bar.pushButtonClose.clicked.connect(self.quit_system)

    def quit_system(self):
        reply = QMessageBox.question(self.title_bar,
                                     '康复手势対侧训练监控系统',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.status = 0
            if hasattr(self, '_vertices_queue'):
                if self._worker_process4.is_alive():
                    self._vertices_queue.put('STOP')
                    while self._output_queue.empty():
                        pass
            if hasattr(self, '_device_input_queue'):
                self._device_input_queue.put('STOP')
            self.delay_quit()

    def delay_getshape(self):
        # Delay 5 seconds for self using
        start_time = time.clock()
        cur_time = start_time
        while cur_time - start_time <= 5.0:
            print("\rCounting down %ds" % int(5.0 - (cur_time - start_time)), end="")
            cur_time = time.clock()
        self.delay_complete_signal.emit()

    def delay_quit(self):
        start_time = time.clock()
        cur_time = start_time
        while cur_time - start_time <= 0.5:
            cur_time = time.clock()
        # Clear queues
        # self._worker_process4 = Process(target=mesh_displayer.Worker, args=(self._vertices_queue, self._output_queue, 4, window_size, ks))
        # self.release_queues()
        self.close_externalEXE()
        self.close()
        # sys.exit()
        os._exit(0)
        # PROCNAME = "python.exe"
        # for proc in psutil.process_iter():
        #     if proc.name() == PROCNAME:
        #         proc.kill()

    def release_queues(self):
        # Process-1
        if hasattr(self, '_device_input_queue'):
            for i in range(self._device_input_queue.qsize()):
                self._device_input_queue.get()
        if hasattr(self, '_device_output_queue'):
            for i in range(self._device_output_queue.qsize()):
                self._device_output_queue.get()
        # Process-estimate
        if hasattr(self, '_estimate_output_queue'):
            for i in range(self._estimate_output_queue.qsize()):
                self._estimate_output_queue.get()
        # Process-display
        if hasattr(self, '_vertices_queue'):
            for i in range(self._vertices_queue.qsize()):
                self._vertices_queue.get()
        if hasattr(self, '_output_queue'):
            for i in range(self._output_queue.qsize()):
                self._output_queue.get()
        
    def on_RegistrationUICancel_clicked(self):
        self.status = 0
        if hasattr(self, '_vertices_queue'):
            if self._worker_process4.is_alive():
                self._vertices_queue.put('STOP')
                while self._output_queue.empty():
                    pass
        if hasattr(self, '_device_output_queue'):
            self._device_output_queue.put('STOP')
        self.registration_UI.on_Cancel_clicked()
    
    def on_RegistrationUIConfirm_clicked(self):
        self.status = 0
        if hasattr(self, '_vertices_queue'):
            if self._worker_process4.is_alive():
                self._vertices_queue.put('STOP')
                while self._output_queue.empty():
                    pass
        if hasattr(self, '_device_output_queue'):
            self._device_output_queue.put('STOP')
        self.registration_UI.on_Confirm_clicked()
    
    def on_getElectricityThresholdStart_clicked(self):
        self.signin_UI.sickside = "left" if self.registration_UI.sickside == "左" else "right"
        # self.on_singlecam_estimate_triggered(self.getshape_opt_params, self.getshape_hand_joints, self.getshape_extra_verts, send_angles=False)
        self.on_switchGesture_clicked(first=True)
    
    def on_getElectricityThresholdCapture_clicked(self):
        if self.ui.gesture not in self.registration_UI.user_elecThreshold_dict:
            self.registration_UI.user_elecThreshold_dict[self.ui.gesture] = dict()
        if self.ui.sickside_angle.text() not in self.registration_UI.user_elecThreshold_dict[self.ui.gesture]:
            self.registration_UI.user_elecThreshold_dict[self.ui.gesture][self.ui.sickside_angle.text()] = dict()
        for i in range(1, 9):
            self.registration_UI.user_elecThreshold_dict[self.ui.gesture][self.ui.sickside_angle.text()]["%d" % i] = \
                getattr(self.ui, "electricity_{}".format(i)).text()
        print(self.registration_UI.user_elecThreshold_dict)
    
    def on_switchGesture_clicked(self, first=False):
        if not first:
            self._device_output_queue.put('STOP')
        height = int(self.settings.value("CAMERA/HEIGHT"))
        width = int(self.settings.value("CAMERA/WIDTH"))
        detect_threshold = float(self.settings.value("OPTIMIZATION/DETECT_THRESHOLD"))
        detect_inputsize = int(self.settings.value("OPTIMIZATION/DETECT_INPUTSIZE"))
        pose_inputsize = int(self.settings.value("OPTIMIZATION/POSE_INPUTSIZE"))
        verbose = True
        detectmodel_path = self.settings.value("PATH/DETECTOR_MODEL")
        step_size = float(self.settings.value("OPTIMIZATION/STEP_SIZE"))
        num_iters = int(self.settings.value("OPTIMIZATION/NUM_ITERS"))
        threshold = float(self.settings.value("OPTIMIZATION/THRESHOLD"))
        lefthand = self.settings.value("PATH/LEFT_MANO")
        righthand = self.settings.value("PATH/RIGHT_MANO")
        w_silhouette = float(self.settings.value("OPTIMIZATION/W_SILHOUETTE"))
        w_pointcloud = float(self.settings.value("OPTIMIZATION/W_POINTCLOUD"))
        w_poseprior = float(self.settings.value("OPTIMIZATION/W_POSEPRIOR"))
        w_shapeprior = float(self.settings.value("OPTIMIZATION/W_SHAPEPRIOR"))
        w_reprojection = float(self.settings.value("OPTIMIZATION/W_REPROJECTION"))
        use_pcaprior = bool(int(self.settings.value("OPTIMIZATION/USE_PCAPRIOR")))
        all_pose = bool(int(self.settings.value("OPTIMIZATION/ALL_POSE")))
        opt_method = int(self.settings.value("OPTIMIZATION/METHOD"))
        # vis_method = self.settings.value("VISUALIZATION/METHOD")

        self._prepare_output_queue = mp.Queue()
        self._prepare_input_queue = mp.Queue()
        self._prepare_process = mp.Process(target=Prepare_Worker, args=(self._prepare_input_queue,
                                                                        self._prepare_output_queue, 2,
                                                                        height, width,
                                                                        detect_threshold, detect_inputsize, pose_inputsize,
                                                                        verbose, detectmodel_path, step_size, num_iters,
                                                                        threshold, lefthand, righthand, w_silhouette,
                                                                        w_pointcloud, w_poseprior, w_shapeprior, w_reprojection,
                                                                        use_pcaprior, self.getshape_opt_params, "prepare",
                                                                        all_pose, opt_method))
        self._prepare_input_queue.put(self.inputs)
        self._prepare_process.start()
        # res_thread = Thread(target=self.get_prepare_result, args=(vis_method,))
        res_thread = Thread(target=self.get_switch_result, args=())
        res_thread.setDaemon(True)
        res_thread.start()
    
    def on_gethandshape_clicked(self):
        delay_thread = Thread(target=self.delay_getshape, args=())
        delay_thread.setDaemon(True)
        delay_thread.start()

    def on_gethandshape(self):

        height = int(self.settings.value("CAMERA/HEIGHT"))
        width = int(self.settings.value("CAMERA/WIDTH"))
        detect_threshold = float(self.settings.value("OPTIMIZATION/DETECT_THRESHOLD"))
        detect_inputsize = int(self.settings.value("OPTIMIZATION/DETECT_INPUTSIZE"))
        pose_inputsize = int(self.settings.value("OPTIMIZATION/POSE_INPUTSIZE"))
        verbose = True
        detectmodel_path = self.settings.value("PATH/DETECTOR_MODEL")
        step_size = float(self.settings.value("OPTIMIZATION/STEP_SIZE"))
        num_iters = int(self.settings.value("OPTIMIZATION/NUM_ITERS"))
        threshold = float(self.settings.value("OPTIMIZATION/THRESHOLD"))
        lefthand = self.settings.value("PATH/LEFT_MANO")
        righthand = self.settings.value("PATH/RIGHT_MANO")
        w_silhouette = float(self.settings.value("OPTIMIZATION/W_SILHOUETTE"))
        w_pointcloud = float(self.settings.value("OPTIMIZATION/W_POINTCLOUD"))
        w_poseprior = float(self.settings.value("OPTIMIZATION/W_POSEPRIOR"))
        w_shapeprior = float(self.settings.value("OPTIMIZATION/W_SHAPEPRIOR"))
        w_reprojection = float(self.settings.value("OPTIMIZATION/W_REPROJECTION"))
        use_pcaprior = bool(int(self.settings.value("OPTIMIZATION/USE_PCAPRIOR")))
        all_pose = bool(int(self.settings.value("OPTIMIZATION/ALL_POSE")))
        opt_method = int(self.settings.value("OPTIMIZATION/METHOD"))
        vis_method = self.settings.value("VISUALIZATION/METHOD")  # 可视化方法, open3d或mayavi

        if not hasattr(self, '_device_output_queue'):
            print("相机未打开，请先打开相机！")
            return

        self._getshape_output_queue = mp.Queue()
        self._getshape_input_queue = mp.Queue()
        self._getshape_process = mp.Process(target=Prepare_Worker, args=(self._getshape_input_queue,
                                                                         self._getshape_output_queue, 2,
                                                                         height, width,
                                                                         detect_threshold, detect_inputsize, pose_inputsize,
                                                                         verbose, detectmodel_path, step_size, num_iters,
                                                                         threshold, lefthand, righthand, w_silhouette,
                                                                         w_pointcloud, w_poseprior, w_shapeprior, w_reprojection,
                                                                         use_pcaprior, None, "getshape", all_pose, opt_method))
        self._getshape_input_queue.put(self.inputs)
        self._getshape_process.start()
        # delay_thread = Thread(target=self.delay_some, args=(self._getshape_process, self._getshape_input_queue))
        # delay_thread.setDaemon(True)
        # delay_thread.start()
        res_thread = Thread(target=self.getshape_result, args=(vis_method,))
        res_thread.setDaemon(True)
        res_thread.start()

    def getshape_result(self, vis_method):
        while True:
            if self._getshape_output_queue.qsize() > 0:
                getshape_result = self._getshape_output_queue.get()
                if type(getshape_result) == list:
                    print("Can't find two hands!")
                    # continue
                break

        self.getshape_opt_params = getshape_result['opt_params']
        vertices = getshape_result['vertices']
        self.getshape_hand_joints = getshape_result['hand_joints']
        self.getshape_extra_verts = getshape_result['extra_verts']
        self.registration_UI.update_handshape(self.getshape_opt_params)
        self.start_mesh_display_process()
        self.update_display3d(vertices)
        # time.sleep(1.0)
        # self.quit_mesh_displayer()
        self.registration_UI.confirm.clicked.connect(self.quit_mesh_displayer)
        self.registration_UI.cancel.clicked.connect(self.quit_mesh_displayer)

    def quit_mesh_displayer(self):
        self._vertices_queue.put("STOP")

    def on_doublecam_prepare_triggered(self):
        # 基于双相机的手势姿态估计
        # TODO
        pass

    def on_doublecam_terminate_triggered(self):
        # TODO
        pass

    def get_switch_result(self):
        while True:
            if self._prepare_output_queue.qsize() > 0:
                prepare_result = self._prepare_output_queue.get()
                break
        if type(prepare_result) == dict:
            opt_params = prepare_result['opt_params']
            vertices = prepare_result['vertices']
            self.hand_joints = prepare_result['hand_joints']
            self.extra_verts = prepare_result['extra_verts']
            # self.start_mesh_display_process()
            self.update_display3d(vertices)
            self.on_singlecam_estimate_triggered(opt_params, self.hand_joints, self.extra_verts)
        else:
            # 检测到的手少于2
            QMessageBox.warning(self, "警告", "检测到%d只手, 不足2" % prepare_result, QMessageBox.Yes)

    # def get_prepare_result(self, vis_method):
    def get_prepare_result(self):
        while True:
            if self._prepare_output_queue.qsize() > 0:
                prepare_result = self._prepare_output_queue.get()
                break
        if type(prepare_result) == dict:
            opt_params = prepare_result['opt_params']
            vertices = prepare_result['vertices']
            self.hand_joints = prepare_result['hand_joints']
            self.extra_verts = prepare_result['extra_verts']
            self.signin_UI.update_initpose(opt_params)
            # if vis_method == "open3d":
            self.start_mesh_display_process()
            # self.register_initjoints(hand_joints)
            self.update_display3d(vertices)
            # else:
            #     focalpoint = (vertices[0, :] + vertices[778, :]) / 2
            #     self.ui.mayavi_widget.update_vertices(vertices, focalpoint)
            # self.on_hrnet_triggered(hand_joints)
            self.on_singlecam_estimate_triggered(opt_params, self.hand_joints, self.extra_verts)
        else:
            # 检测到的手少于2
            QMessageBox.warning(self, "警告", "检测到%d只手, 不足2" % prepare_result, QMessageBox.Yes)
    
    def on_singlecam_prepare_triggered(self):
        delay_thread = Thread(target=self.delay_prepare, args=())
        delay_thread.setDaemon(True)
        delay_thread.start()
    
    def delay_prepare(self):
        # Delay 5 seconds for self using
        start_time = time.clock()
        cur_time = start_time
        while cur_time - start_time <= 5.0:
            print("\rCounting down %ds" % int(5.0 - (cur_time - start_time)), end="")
            cur_time = time.clock()
        self.delay_prepare_signal.emit()

    def on_singlecam_prepare(self):
        # print('self.signin_UI.dict_elec_threshold', self.signin_UI.dict_elec_threshold)
        # self.elec_threshold_dict = self.signin_UI.dict_elec_threshold[self.ui.gesture]
        # print(self.elec_threshold_dict.keys())
        # kwargs
        model = self.settings.value("OPTIMIZATION/MODEL")
        if model == 'TemporalSmoothing':
            self.on_singlecam_estimate_triggered(None, None, None)
        elif model =='HandTailor':
            self.on_singlecam_estimate_triggered(None, None, None)
        return

        est_method = int(self.settings.value("OPTIMIZATION/EST_METHOD"))
        if self.num_start_triggered == 0 or est_method == 0:
            height = int(self.settings.value("CAMERA/HEIGHT"))
            width = int(self.settings.value("CAMERA/WIDTH"))
            detect_threshold = float(self.settings.value("OPTIMIZATION/DETECT_THRESHOLD"))
            detect_inputsize = int(self.settings.value("OPTIMIZATION/DETECT_INPUTSIZE"))
            pose_inputsize = int(self.settings.value("OPTIMIZATION/POSE_INPUTSIZE"))
            verbose = True
            detectmodel_path = self.settings.value("PATH/DETECTOR_MODEL")
            step_size = float(self.settings.value("OPTIMIZATION/STEP_SIZE"))
            num_iters = int(self.settings.value("OPTIMIZATION/NUM_ITERS"))
            threshold = float(self.settings.value("OPTIMIZATION/THRESHOLD"))
            lefthand = self.settings.value("PATH/LEFT_MANO")
            righthand = self.settings.value("PATH/RIGHT_MANO")
            w_silhouette = float(self.settings.value("OPTIMIZATION/W_SILHOUETTE"))
            w_pointcloud = float(self.settings.value("OPTIMIZATION/W_POINTCLOUD"))
            w_poseprior = float(self.settings.value("OPTIMIZATION/W_POSEPRIOR"))
            w_shapeprior = float(self.settings.value("OPTIMIZATION/W_SHAPEPRIOR"))
            w_reprojection = float(self.settings.value("OPTIMIZATION/W_REPROJECTION"))
            use_pcaprior = bool(int(self.settings.value("OPTIMIZATION/USE_PCAPRIOR")))
            all_pose = bool(int(self.settings.value("OPTIMIZATION/ALL_POSE")))
            opt_method = int(self.settings.value("OPTIMIZATION/METHOD"))
            # vis_method = self.settings.value("VISUALIZATION/METHOD")

            self._prepare_output_queue = mp.Queue()
            self._prepare_input_queue = mp.Queue()
            self._prepare_process = mp.Process(target=Prepare_Worker, args=(self._prepare_input_queue,
                                                                            self._prepare_output_queue, 2,
                                                                            height, width,
                                                                            detect_threshold, detect_inputsize, pose_inputsize,
                                                                            verbose, detectmodel_path, step_size, num_iters,
                                                                            threshold, lefthand, righthand, w_silhouette,
                                                                            w_pointcloud, w_poseprior, w_shapeprior, w_reprojection,
                                                                            use_pcaprior, self.signin_UI.init_params, "prepare",
                                                                            all_pose, opt_method))
            self._prepare_input_queue.put(self.inputs)
            self._prepare_process.start()
            # res_thread = Thread(target=self.get_prepare_result, args=(vis_method,))
            res_thread = Thread(target=self.get_prepare_result, args=())
            res_thread.setDaemon(True)
            res_thread.start()
        else:
            self.start_mesh_display_process()
            self.update_display3d(self.vertices)
            self.on_singlecam_estimate_triggered(self.opt_params, self.hand_joints, self.extra_verts)

        self.num_start_triggered += 1

    def on_envoke_exteranl(self, exe_file):
        if os.path.exists(exe_file):
            self._external_process = mp.Process(target=External_Worker, args=(exe_file, 5))
            self._external_process.start()
            print("External process {} start.".format(exe_file.split('/')[-1]))
        else:
            QMessageBox.warning(self, "警告", "外部程序不存在", QMessageBox.Yes)
    
    def close_externalEXE(self):
        PROCNAME = self.settings.value("PATH/EXE_FILE").split('/')[-1]
        for proc in psutil.process_iter():
            if proc.name() == PROCNAME:
                proc.kill()

    def on_singlecam_terminate_triggered(self):
        # 终止康复手势的估计
        # if self._prepare_process.is_alive():
        #     print("Prepare process is still alive!")
        #     return
        if self._estimate_process.is_alive():
            self._device_output_queue.put("STOP")
        if hasattr(self, '_worker_process4') and self._worker_process4.is_alive():
            self._vertices_queue.put("STOP")
        if hasattr(self, 'goodside_angle') and hasattr(self, 'goodside_angle'):
            del self.goodside_angle
            del self.sickside_angle
        # self.ui.round_progresser.Reset()

    def on_singlecam_estimate_triggered(self, init_params, hand_joints, extra_verts, send_angles=True):
        # if self.wrist_loc_l is None or self.wrist_loc_r is None or self.y_mtip is None:
        #     QMessageBox.warning(self, "警告", "请先准备, 再点击开始", QMessageBox.Yes)
        #     return
        # else:
        model = self.settings.value("OPTIMIZATION/MODEL")
        self.elec = 0
        if model == 'TemporalSmoothing':
            # TemporalSmoothing

            match_threshold = float(self.settings.value("MONITOR/MATCH_THRESHOLD"))
            angle_length = int(self.settings.value("MONITOR/ANGLE_LENGTH"))
            left = True if self.signin_UI.sickside == "left" else False

            self._estimate_output_queue = mp.Queue()
            self._estimate_process = mp.Process(target=Estimate_TemporalSmoothing_Worker,
                                                args=(self._device_output_queue,
                                                      self._estimate_output_queue,
                                                      self.ui.gesture,
                                                      left))

            self._estimate_process.start()
            self.elecState = 0
            self.goodside_angle_queue = deque(maxlen=angle_length)
            self.sickside_angle_queue = deque(maxlen=angle_length)

            est_thread = Thread(target=self.update_estimate, args=(match_threshold, send_angles))
            est_thread.setDaemon(True)
            est_thread.start()
        elif model == 'HandTailor':
            # HandTailor
            match_threshold = float(self.settings.value("MONITOR/MATCH_THRESHOLD"))
            angle_length = int(self.settings.value("MONITOR/ANGLE_LENGTH"))
            left = True if self.signin_UI.sickside == "left" else False

            self._estimate_output_queue = mp.Queue()
            self._estimate_process = mp.Process(target=Estimate_HandTailor_Worker, args=(self._device_output_queue,
                                                                                         self._estimate_output_queue,
                                                                                         self.ui.gesture,
                                                                                         left))

            self._estimate_process.start()
            self.elecState = 0
            self.goodside_angle_queue = deque(maxlen=angle_length)
            self.sickside_angle_queue = deque(maxlen=angle_length)

            self.start_mesh_display_process()
            est_thread = Thread(target=self.update_estimate, args=(match_threshold, send_angles))
            est_thread.setDaemon(True)
            est_thread.start()
        return

        height = int(self.settings.value("CAMERA/HEIGHT"))
        width = int(self.settings.value("CAMERA/WIDTH"))
        step_size = float(self.settings.value("OPTIMIZATION/STEP_SIZE_2"))
        num_iters = int(self.settings.value("OPTIMIZATION/NUM_ITERS_2"))
        threshold = float(self.settings.value("OPTIMIZATION/THRESHOLD"))
        lefthand = self.settings.value("PATH/LEFT_MANO")
        righthand = self.settings.value("PATH/RIGHT_MANO")
        w_silhouette = float(self.settings.value("OPTIMIZATION/W_SILHOUETTE"))
        w_pointcloud = float(self.settings.value("OPTIMIZATION/W_POINTCLOUD"))
        w_poseprior = float(self.settings.value("OPTIMIZATION/W_POSEPRIOR"))
        w_reprojection = float(self.settings.value("OPTIMIZATION/W_REPROJECTION"))
        w_temporalprior = float(self.settings.value("OPTIMIZATION/W_TEMPORALPRIOR"))
        wrist_rot_pitch = float(self.settings.value("MONITOR/WRIST_ROT_PITCH"))
        wrist_rot_yaw = float(self.settings.value("MONITOR/WRIST_ROT_YAW"))
        use_pcaprior = bool(int(self.settings.value("OPTIMIZATION/USE_PCAPRIOR")))
        match_threshold = float(self.settings.value("MONITOR/MATCH_THRESHOLD"))
        angle_length = int(self.settings.value("MONITOR/ANGLE_LENGTH"))
        # vis_method = self.settings.value("VISUALIZATION/METHOD")
        left = True if self.signin_UI.sickside == "left" else False

        self._estimate_output_queue = mp.Queue()
        self._estimate_process = mp.Process(target=Estimate_Worker, args=(self._device_output_queue,
                                                                          self._estimate_output_queue, 8, init_params, hand_joints, extra_verts,
                                                                          wrist_rot_pitch, wrist_rot_yaw, self.ui.gesture,
                                                                          # self.wrist_loc_l, self.wrist_loc_r, self.y_mtip,
                                                                          height, width, step_size, num_iters,
                                                                          threshold, lefthand, righthand,
                                                                          w_silhouette, w_pointcloud, w_poseprior, w_reprojection, w_temporalprior,
                                                                          left, use_pcaprior))

        self._estimate_process.start()
        self.elecState = 0
        self.goodside_angle_queue = deque(maxlen=angle_length)
        self.sickside_angle_queue = deque(maxlen=angle_length)

        # self.start_mesh_display_process()
        est_thread = Thread(target=self.update_estimate, args=(match_threshold, send_angles))
        est_thread.setDaemon(True)
        est_thread.start()

    def update_estimate(self, match_threshold, send_angles=True):
        self.start_time = time.time()
        follow_type = self.settings.value("MONITOR/FOLLOW_TYPE")
        while True:
            if not self._estimate_output_queue.empty():
                meta = self._estimate_output_queue.get()
                if type(meta) == dict:
                    completeness = meta['completeness']
                    self.vertices = meta['vertices']
                    angles = meta['angles']
                    self.ui.sickside_angle.setText(str(angles[0]))
                    self.ui.goodside_angle.setText(str(angles[1]))
                    mismatchness = meta['mismatchness']
                    self.hand_joints = meta['hand_joints']
                    self.opt_params = meta["opt_params"]
                    self.ui.electricity_1.setText(str(self.elec))
                    if send_angles:
                        self.goodside_angle_queue.append(angles[1])
                        self.sickside_angle_queue.append(angles[0])
                        if follow_type == "A":
                            self.socket_send(angles)
                        elif follow_type == "B":
                            self.socket_send_by_sickside(angles)
                        elif follow_type == "C":
                            self.socket_send_by_sickside_rt(angles)
                        elif follow_type == "D":
                            self.socket_send_by_sickside_D(angles)
                    if self.settings.value("OPTIMIZATION/MODEL") == 'HandTailor':
                        self.update_display3d(self.vertices)


    def register_initjoints(self, initjoints):
        if initjoints.shape[0] == 2:
            if initjoints[1][9, 0] > initjoints[0][9, 0]:
                self.wrist_loc_l = initjoints[1][9, :]
                self.wrist_loc_r = initjoints[0][9, :]
            else:
                self.wrist_loc_l = initjoints[0][9, :]
                self.wrist_loc_r = initjoints[1][9, :]
            self.y_mtip = np.max(initjoints[:, :, 1])
        elif initjoints.shape[0] < 2:
            QMessageBox.warning(self, "警告", "手部检测数少于2", QMessageBox.Yes)
            self.wrist_loc_l = None
            self.wrist_loc_r = None
            self.y_mtip = None
        else:
            QMessageBox.warning(self, "警告", "手部检测数超过2", QMessageBox.Yes)
            self.wrist_loc_l = None
            self.wrist_loc_r = None
            self.y_mtip = None

    def update_display3d(self, vertices):
        """
        Parameters:
        -----------
        vertices: vertices of mano hand model, 778 x 3
        """
        self._vertices_queue.put(vertices)
        # self.ui.mayavi_widget.update_vertices(vertices)

    # def update_completeness(self, parameters):
    #     """
    #     Parameters:
    #     -----------
    #     parameters: pose parameters or quat parameters
    #     """
    #     if self.signin_UI.sickside == "left":
    #         completeness = self.completeness_estimator(parameters, self.init_params, left=True)
    #     else:
    #         completeness = self.completeness_estimator(parameters, self.init_params, left=False)
    #     assert completeness >= 0, "Invalid completeness {}".format(completeness)
    #     # update
    #     self.ui.round_progresser.UpdatePercent(completeness)

    def on_action_fist_triggered(self):
        # self.ui.round_progresser.UpdateGesture("握拳")
        self.ui.display.setText("握拳")

    def on_action_wristforward_triggered(self):
        self.ui.display.setText("腕前屈")

    def on_action_wristbackward_triggered(self):
        # self.ui.round_progresser.UpdateGesture("腕背伸")
        self.ui.display.setText("腕背伸")

    def on_action_wristside_triggered(self):
        # self.ui.round_progresser.UpdateGesture("尺偏&桡偏")
        self.ui.display.setText("腕尺桡偏")

    def on_preference_apply(self):
        # Camera
        self.settings.setValue("CAMERA/WIDTH", self.preference_UI.width_edit.text())
        self.settings.setValue("CAMERA/HEIGHT", self.preference_UI.height_edit.text())
        self.settings.setValue("CAMERA/FPS", self.preference_UI.fps_edit.text())
        self.settings.setValue("CAMERA/THRESHOLD", self.preference_UI.threshold_edit.text())
        # Optimization
        self.settings.setValue("OPTIMIZATION/STEP_SIZE", self.preference_UI.stepsize_1_edit.text())
        self.settings.setValue("OPTIMIZATION/NUM_ITERS", self.preference_UI.numiters_1_edit.text())
        self.settings.setValue("OPTIMIZATION/STEP_SIZE_2", self.preference_UI.stepsize_2_edit.text())
        self.settings.setValue("OPTIMIZATION/NUM_ITERS_2", self.preference_UI.numiters_2_edit.text())
        self.settings.setValue("OPTIMIZATION/METHOD", self.preference_UI.method.currentText())
        self.settings.setValue("OPTIMIZATION/EST_METHOD", self.preference_UI.est_method.currentText())
        self.settings.setValue("OPTIMIZATION/ALL_POSE", 1 if self.preference_UI.all_pose.isChecked() else 0)
        self.settings.setValue("OPTIMIZATION/USE_PCAPRIOR", 1 if self.preference_UI.use_prior.isChecked() else 0)
        self.settings.setValue("OPTIMIZATION/THRESHOLD", self.preference_UI.optim_threshold_edit.text())
        self.settings.setValue("OPTIMIZATION/W_SILHOUETTE", self.preference_UI.w_silhouette_edit.text())
        self.settings.setValue("OPTIMIZATION/W_POINTCLOUD", self.preference_UI.w_pointcloud_edit.text())
        self.settings.setValue("OPTIMIZATION/W_POSEPRIOR", self.preference_UI.w_poseprior_edit.text())
        self.settings.setValue("OPTIMIZATION/W_TEMPORALPRIOR", self.preference_UI.w_temporalprior_edit.text())
        self.settings.setValue("OPTIMIZATION/W_SHAPEPRIOR", self.preference_UI.w_shapeprior_edit.text())
        self.settings.setValue("OPTIMIZATION/MODEL", self.preference_UI.model.currentText())
        # Monitor
        self.settings.setValue("MONITOR/CHECK_FPS", self.preference_UI.check_fps_edit.text())
        self.settings.setValue("MONITOR/COMPLETE_TYPE", self.preference_UI.complete_box.currentText())
        self.settings.setValue("VISUALIZATION/METHOD", self.preference_UI.vis_box.currentText())
        self.settings.setValue("MONITOR/WRIST_ROT_PITCH", self.preference_UI.wrist_pitch_edit.text())
        self.settings.setValue("MONITOR/WRIST_ROT_YAW", self.preference_UI.wrist_yaw_edit.text())
        self.settings.setValue("MONITOR/MATCH_THRESHOLD", self.preference_UI.match_threshold_edit.text())
        # Path
        self.settings.setValue("PATH/LEFT_MANO", self.preference_UI.lefthand_edit.text())
        self.settings.setValue("PATH/RIGHT_MANO", self.preference_UI.righthand_edit.text())
        self.settings.setValue("PATH/EXE_FILE", self.preference_UI.device_exe.text())
        self.settings.setValue("PATH/DETECTOR_MODEL", self.preference_UI.detector_edit.text())
        self.settings.setValue("PATH/POSE_MODEL", self.preference_UI.pose_edit.text())

    def on_preference_confirm(self):
        # Update subclass
        # self.camera = camera.RealsenseCapture(self.settings)
        # self.camera.display_image_signal.connect(self.PutImg)
        self.preference_UI.close()
    
    def on_export_triggered(self):
        filename, _ = QFileDialog.getSaveFileName(self,
                                                  "导出",
                                                  os.path.join(os.path.expanduser("~"), "Desktop"),
                                                  "CSV Files (*.csv)")
        if not filename.endswith('.csv'):
            filename = filename + '.csv'
        con = sqlite3.connect(sqlite_data)
        table = sql.read_sql('select * from data', con)
        table.to_csv(filename)
    
    def message_received(self, client, server, message):
        data_dict = json.loads(message)
        if isinstance(data_dict, str):
            data_dict = json.loads(data_dict)
        if 'command' in data_dict:
            self.ui.ElectrodeState.setText(data_dict['command'])
            if data_dict['command'] == 'ElectrodeState':
                self.ui.state.setText(data_dict['state'])
            elif data_dict['command'] == 'Electric':
                channel = int(data_dict['channel'])
                getattr(self.ui, "electricity_{}".format(channel + 1)).setText(data_dict['electricity'])
                # self.ui.state.setText(data_dict['electricity']) # Channel
        # print("Client {} said: {}".format(client['id'], message))
    
    def get_electricity_command_func(self, command, channel):

        def func():
            message_dict = {"Channel": str(channel), "command": command}
            message = str(message_dict).replace("'", "\"")
            self.server.send_message_to_all(message)
        
        return func
    
    def on_StopEmergency_Clicked(self):
        self.server.send_message_to_all("Stop")
    
    def evoke_socket(self):
        from websocket_server import WebsocketServer
        self.server = WebsocketServer(int(self.settings.value("SOCKET/PORT")), host="localhost")
        self.server.set_fn_message_received(self.message_received)
        print("服务器启动成功")
        est_thread = Thread(target=self.server.run_forever, args=())
        est_thread.setDaemon(True)
        est_thread.start()
    
    def update_recv_ws(self):
        # self.socket = websocket.create_connection("ws://localhost:{}".format(self.settings.value("SOCKET/PORT")))
        # while True:
        #    response = self.socket.recv()
        #    self.server.send("1000")
        #    if response:
        #        print(response)
        pass

    def find_matchest_angle(self, angle):
        recommand_electricity_list = [int(item) for item in self.elec_threshold_dict.keys()]
        recommand_electricity_list.sort()
        # print(recommand_electricity_list)
        for i, v in enumerate(recommand_electricity_list):
            if i == 0:
                if angle < v:
                    # message = self.elec_threshold_dict[str(v)]
                    message = {'1': '0', '2': '0', '3': '0', '4': '0', '5': '0', '6': '0', '7': '0', '8': '0'}
                    message["command"] = "recommandElectricity"
                    return message
            elif i == (len(recommand_electricity_list) - 1):
                if angle > v:
                    message = self.elec_threshold_dict[str(v)]
                    message["command"] = "recommandElectricity"
                    return message
            else:
                if angle > v and angle <= recommand_electricity_list[i + 1]:
                    message = self.elec_threshold_dict[recommand_electricity_list[i + 1]]
                    message["command"] = "recommandElectricity"
                    return message
    
    def get_max_angle_elec(self):
        # print('self.signin_UI.dict_elec_threshold', self.signin_UI.dict_elec_threshold)

        # recommand_electricity_list = [int(item) for item in self.elec_threshold_dict.keys()]
        # recommand_electricity_list.sort()
        # max_angle = recommand_electricity_list[-1]
        max_angle = 10
        # message = self.elec_threshold_dict[str(max_angle)]
        message = {"command": "MaxElectricity"}
        return message

    def Reduce(self):
        message_dict = {"Channel": "0", "command": "Reduce"}
        message = str(message_dict).replace("'", "\"")
        self.server.send_message_to_all(message)
        print(message)

    def Increase(self):
        message_dict = {"Channel": "0", "command": "Increase"}
        message = str(message_dict).replace("'", "\"")
        self.server.send_message_to_all(message)
        print(message)

    def socket_send_by_sickside_D(self, angles):
        sickside_angle, goodside_angle = angles[0], angles[1]
        if self.elecState == 0:
            print("Start electric.")
            self.elec = 0
            self.server.send_message_to_all("Start")
            self.server.send_message_to_all(str(self.get_max_angle_elec()).replace("'", "\""))
            self.elecState = 1
        if time.time() - self.start_time >= float(self.settings.value("MONITOR/SECOND_PER_SEND")):
            threshold = int(self.settings.value("MONITOR/ANGLE_THRESHOLD")) * (1 if self.ui.gesture=='fist' else 2)
            if abs(sickside_angle - goodside_angle) <= threshold:
                return
            if hasattr(self, 'sickside_angle'):
                if goodside_angle - sickside_angle <= threshold and self.elec > 0:
                    self.Reduce()
                    self.elec -= 1

                elif goodside_angle - sickside_angle < self.goodside_angle - self.sickside_angle - threshold and self.elec > 0:
                    self.Reduce()
                    self.elec -= 1

                elif goodside_angle - sickside_angle >= self.goodside_angle - self.sickside_angle + threshold and self.elec < 15:
                    self.Increase()
                    self.elec += 1

            else:
                self.goodside_angle = goodside_angle
                self.sickside_angle = sickside_angle


    def socket_send_by_sickside_rt(self, angles):
        sickside_angle, goodside_angle = angles[0], angles[1]
        np_angle_dist = np.array(self.goodside_angle_queue) - np.array(self.sickside_angle_queue)
        if self.elecState == 0 and len(np_angle_dist) >= int(self.settings.value("MONITOR/STAY_LENGTH")):
            # detect movement of goodside first time
            if np_angle_dist[-1] > int(self.settings.value("MONITOR/ANGLE_THRESHOLD")) and \
                np_angle_dist[-int(self.settings.value("MONITOR/STAY_LENGTH"))] > int(self.settings.value("MONITOR/ANGLE_THRESHOLD")):
                self.server.send_message_to_all("Start")
                if hasattr(self, "elec_threshold_dict"):
                    self.server.send_message_to_all(str(self.find_matchest_angle(goodside_angle)).replace("'", "\""))
                self.elecState = 1
        elif self.elecState == 1:
            if abs(sickside_angle - goodside_angle) >= int(self.settings.value("MONITOR/ANGLE_THRESHOLD")):
                if hasattr(self, "elec_threshold_dict"):
                    self.server.send_message_to_all(str(self.find_matchest_angle(goodside_angle)).replace("'", "\""))
            else:
                pass
        
        message_dict = {"gesture": self.ui.gesture,
                        "sickside_angle": sickside_angle,
                        "goodside_angle": goodside_angle}
        message = str(message_dict).replace("'", "\"")
        if time.time() - self.start_time >= 0.07:
            self.server.send_message_to_all(message)
            self.start_time = time.time()
    
    def socket_send_by_sickside(self, angles):
        sickside_angle, goodside_angle = angles[0], angles[1]
        np_angle_dist = np.array(self.goodside_angle_queue) - np.array(self.sickside_angle_queue)
        if self.elecState == 0 and len(np_angle_dist) >= int(self.settings.value("MONITOR/STAY_LENGTH")):
            if np_angle_dist[-1] > int(self.settings.value("MONITOR/ANGLE_THRESHOLD")) and \
                np_angle_dist[-int(self.settings.value("MONITOR/STAY_LENGTH"))] > int(self.settings.value("MONITOR/ANGLE_THRESHOLD")):
                self.server.send_message_to_all("Start")
                if hasattr(self, 'elec_threshold_dict'):
                    self.server.send_message_to_all(str(self.find_matchest_angle(goodside_angle)).replace("'", "\""))
                self.elecState = 1
        elif self.elecState == 1 and abs(sickside_angle - goodside_angle) <= int(self.settings.value("MONITOR/ANGLE_THRESHOLD")):
            self.elecState = 0
            self.goodside_angle_queue.clear()
            self.sickside_angle_queue.clear()
        
        message_dict = {"gesture": self.ui.gesture,
                        "sickside_angle": sickside_angle,
                        "goodside_angle": goodside_angle}
        message = str(message_dict).replace("'", "\"")
        if time.time() - self.start_time > 0.38:
            self.server.send_message_to_all(message)
            self.start_time = time.time()
    
    def socket_send(self, angles):
        # gesture = self.gestures.index(self.ui.gesture)
        sickside_angle, goodside_angle = angles[0], angles[1]
        if len(self.goodside_angle_queue) == self.goodside_angle_queue.maxlen and self.elecState == 0:
            np_goodside_angles = np.array(self.goodside_angle_queue)
            np_mask_equal = np.abs(np_goodside_angles - np_goodside_angles[0]) <= int(self.settings.value("MONITOR/ANGLE_THRESHOLD"))
            if len(np_goodside_angles[np_mask_equal]) >= self.goodside_angle_queue.maxlen * 0.8 and goodside_angle > 10:
                self.server.send_message_to_all("Start")
                if hasattr(self, 'elec_threshold_dict'):
                	self.server.send_message_to_all(str(self.find_matchest_angle(goodside_angle)).replace("'", "\""))
                self.elecState = 1  # 从双手平放的状态转换到健侧做手势，患侧平放
        elif self.elecState == 1 and abs(sickside_angle - goodside_angle) <= int(self.settings.value("MONITOR/ANGLE_THRESHOLD")):
            self.elecState = 0
            self.goodside_angle_queue.clear()

        message_dict = {"gesture": self.ui.gesture,
                        "sickside_angle": sickside_angle,
                        "goodside_angle": goodside_angle}
        message = str(message_dict).replace("'", "\"")
        if time.time() - self.start_time > 2.0:
            self.server.send_message_to_all(message)
            self.start_time = time.time()

# import asyncio
# import websockets


# async def echo(websocket, path):
#     async for message in websocket:
#         message = "I got your message: {}".format(message)
#         await websocket.send(message)
#
#
# asyncio.get_event_loop().run_until_complete(websockets.serve(echo, 'localhost', 8765))
# asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    mw.on_envoke_exteranl(mw.settings.value("PATH/EXE_FILE"))
    app.setStyleSheet(load_darkstyle())
    mainFont = app.font()
    mainFont.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(mainFont)
    sys.exit(app.exec_())
