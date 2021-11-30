from PyQt5.QtWidgets import QMainWindow
from . import Settings_ui


class Preference(QMainWindow, Settings_ui.Ui_Form):

    def __init__(self, settings, parent=None):
        super(Preference, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Preference")
        self.settings = settings

        # Camera
        self.width_edit.setText(settings.value("CAMERA/WIDTH"))
        self.height_edit.setText(settings.value("CAMERA/HEIGHT"))
        self.fps_edit.setText(settings.value("CAMERA/FPS"))
        self.threshold_edit.setText(settings.value("CAMERA/THRESHOLD"))
        # Optimization
        self.stepsize_1_edit.setText(settings.value("OPTIMIZATION/STEP_SIZE"))
        self.numiters_1_edit.setText(settings.value("OPTIMIZATION/NUM_ITERS"))
        self.stepsize_2_edit.setText(settings.value("OPTIMIZATION/STEP_SIZE_2"))
        self.numiters_2_edit.setText(settings.value("OPTIMIZATION/NUM_ITERS_2"))
        self.optim_threshold_edit.setText(settings.value("OPTIMIZATION/THRESHOLD"))
        self.w_silhouette_edit.setText(settings.value("OPTIMIZATION/W_SILHOUETTE"))
        self.w_pointcloud_edit.setText(settings.value("OPTIMIZATION/W_POINTCLOUD"))
        self.w_reprojection_edit.setText(settings.value("OPTIMIZATION/W_REPROJECTION"))
        self.w_poseprior_edit.setText(settings.value("OPTIMIZATION/W_POSEPRIOR"))
        self.w_temporalprior_edit.setText(settings.value("OPTIMIZATION/W_TEMPORALPRIOR"))
        self.w_shapeprior_edit.setText(settings.value("OPTIMIZATION/W_SHAPEPRIOR"))
        self.method.setCurrentText(settings.value("OPTIMIZATION/METHOD"))
        self.model.setCurrentText(settings.value("OPTIMIZATION/MODEL"))
        self.est_method.setCurrentText(settings.value("OPTIMIZATION/EST_METHOD"))
        self.use_prior.setCheckable(True)
        self.all_pose.setCheckable(True)
        if int(settings.value("OPTIMIZATION/USE_PCAPRIOR")) == 1:
            self.use_prior.setChecked(True)
        else:
            self.use_prior.setChecked(False)
        if int(settings.value("OPTIMIZATION/ALL_POSE")) == 1:
            self.all_pose.setChecked(True)
        else:
            self.all_pose.setChecked(False)
        # Monitor
        self.check_fps_edit.setText(settings.value("MONITOR/CHECK_FPS"))
        self.vis_box.setCurrentText(settings.value("VISUALIZATION/METHOD"))
        self.wrist_pitch_edit.setText(settings.value("MONITOR/WRIST_ROT_PITCH"))
        self.wrist_yaw_edit.setText(settings.value("MONITOR/WRIST_ROT_YAW"))
        self.match_threshold_edit.setText(settings.value("MONITOR/MATCH_THRESHOLD"))
        # Path
        self.lefthand_edit.setText(settings.value("PATH/LEFT_MANO"))
        self.righthand_edit.setText(settings.value("PATH/RIGHT_MANO"))
        self.device_exe.setText(settings.value("PATH/EXE_FILE"))
        self.detector_edit.setText(settings.value("PATH/DETECTOR_MODEL"))
        self.pose_edit.setText(settings.value("PATH/POSE_MODEL"))

        # Connect
        self.cancel_button.clicked.connect(self.on_cancel_clicked)

    def on_cancel_clicked(self):
        self.close()


