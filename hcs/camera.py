"""

"""
import pyrealsense2 as rs
import cv2
import numpy as np
from threading import Thread, Lock
import multiprocessing as mp
from utils.realsense_device_manager import DeviceManager
from PyQt5.QtCore import QObject, pyqtSignal


# Using Multiprocessing to capture
def Worker(inputs_queue, outputs_queue, proc_id, resolution_width, resolution_height, frame_rate):

    # resolution_width = int(settings.value("CAMERA/WIDTH"))
    # resolution_height = int(settings.value("CAMERA/HEIGHT"))
    # frame_rate = int(settings.value("CAMERA/FPS"))
    device_instrinsics = None

    # Open Devices
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

    device_manager = DeviceManager(rs.context(), rs_config)

    device_manager.enable_all_devices()
    assert(len(device_manager._available_devices) > 0), "No Devices connected."
    # Enable the emitter of the devices
    device_manager.enable_emitter(True)
    # flag_cap = False

    while True:
        # if inputs_queue.empty():
        frames_devices = device_manager.poll_frames()
        if device_instrinsics is None:
            device_instrinsics = device_manager.get_device_intrinsics(frames_devices)
            device_instrinsics = get_intrinsics(device_instrinsics)
        num_devices = len(device_manager._available_devices)
        num_streams = 2
        grid_images = np.zeros((num_devices * resolution_height,
                                num_streams * resolution_width, 3),
                               dtype=np.uint8)
        color_images = []
        depth_images = []
        # print("len(frames_devices)", len(frames_devices))
        for i, (device, frame) in enumerate(frames_devices.items()):

            color_image = np.asarray(frame[rs.stream.color].get_data())
            depth_image = np.asarray(frame[rs.stream.depth].get_data())
            color_images.append(color_image)
            depth_images.append(depth_image)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            device_images = np.hstack((color_image, depth_colormap))
            height_begin = resolution_height * i
            height_end = height_begin + resolution_height
            grid_images[height_begin:height_end, :, :3] = device_images[..., :3]
        output = {"display": grid_images,
                  "color": color_images,
                  "depth": depth_images,
                  "ks": device_instrinsics}
        outputs_queue.put(output)
        # else:
        if not inputs_queue.empty():
            message = inputs_queue.get()
            if message == 'STOP':
                print("Quit Realsense Capture")
                device_manager.disable_streams()
                break
    outputs_queue.put("STOP")


def get_intrinsics(device_instrinsics):
    instrinsics = []
    for serial, frameset in device_instrinsics.items():
        for key, value in frameset.items():
            if key == rs.stream.color:
                ks = np.eye(3)
                ks[0, 0] = value.fx
                ks[1, 1] = value.fy
                ks[0, 2] = value.ppx
                ks[1, 2] = value.ppy
                instrinsics.append(ks)
    return instrinsics

############################################################################


# Using Multithread to do capture
class RealsenseCapture(QObject):
    display_image_signal = pyqtSignal(np.ndarray)
    est_input_signal = pyqtSignal(object)
    open_signal = pyqtSignal()

    def __init__(self, settings):

        super(RealsenseCapture, self).__init__()
        self.resolution_width = int(settings.value("CAMERA/WIDTH"))
        self.resolution_height = int(settings.value("CAMERA/HEIGHT"))
        self.frame_rate = int(settings.value("CAMERA/FPS"))
        self.device_instrinsics = None
        self.lock = Lock()

    def start(self):
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, self.resolution_width, self.resolution_height, rs.format.z16, self.frame_rate)
        rs_config.enable_stream(rs.stream.color, self.resolution_width, self.resolution_height, rs.format.bgr8, self.frame_rate)

        # Use the device manager class to enable the devices and get the frames
        self.device_manager = DeviceManager(rs.context(), rs_config)
        self.device_manager.enable_all_devices()

        assert(len(self.device_manager._available_devices) > 0)

        # Enable the emitter of the devices
        self.device_manager.enable_emitter(True)

        # define thread
        thread = Thread(target=self.update, args=())
        thread.setDaemon(True)
        thread.start()

    def end(self):

        self.device_manager.disable_streams()

    def update(self):

        while 1:
            # Get frames from all the devices
            frames_devices = self.device_manager.poll_frames()
            num_devices = len(self.device_manager._available_devices)
            num_streams = 2
            grid_images = np.zeros((num_devices * self.resolution_height,
                                    num_streams * self.resolution_width, 3),
                                   dtype=np.uint8)
            color_images = []
            depth_images = []
            for i, (device, frame) in enumerate(frames_devices.items()):

                color_image = np.asarray(frame[rs.stream.color].get_data())
                depth_image = np.asarray(frame[rs.stream.depth].get_data())
                color_images.append(color_image)
                depth_images.append(depth_image)

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                device_images = np.hstack((color_image, depth_colormap))
                height_begin = self.resolution_height * i
                height_end = height_begin + self.resolution_height
                grid_images[height_begin:height_end, :, :3] = device_images[..., :3]

            self.display_image_signal.emit(grid_images)
            self.display_images = grid_images
            self.color_images = color_images
            self.depth_images = depth_images
            if self.device_instrinsics is None:
                self.device_instrinsics = self.device_manager.get_device_intrinsics(frames_devices)
                instrinsics = []
                for serial, frameset in self.device_instrinsics.items():
                    for key, value in frameset.items():
                        if key == rs.stream.depth:
                            ks = np.eye(3)
                            ks[0, 0] = value.fx
                            ks[1, 1] = value.fy
                            ks[0, 2] = value.ppx
                            ks[1, 2] = value.ppy
                            instrinsics.append(ks)
                self.ks = instrinsics
                self.open_signal.emit()
            self.est_input_signal.emit({'depth': depth_images,
                                        'color': color_images,
                                        'ks': self.ks})
            

    def read(self):
        """
        Get current captured depth and color images(aligned)

        Return:
        ------------
        depth, color, ks
        """
        # instrinsics = []
        # for serial, frameset in self.device_instrinsics.items():
        #     for key, value in frameset.items():
        #         if key == rs.stream.depth:
        #             ks = np.eye(3)
        #             ks[0, 0] = value.fx
        #             ks[1, 1] = value.fy
        #             ks[0, 2] = value.ppx
        #             ks[1, 2] = value.ppy
        #             instrinsics.append(ks)
        return {'depth': self.depth_images,
                'color': self.color_images,
                'ks': self.ks}

