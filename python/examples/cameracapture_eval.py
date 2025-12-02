from collections import defaultdict
import multiprocessing as mp
import time
import numpy as np
import pickle
import sys
import os
import pyrealsense2 as rs
import cv2
from tqdm import tqdm
from helper import NumpyAccumulator

# --- The Camera Process Class ---
class CameraCaptureProcess(mp.Process):
    def __init__(self, cmd_queue, config, serial_number, obs_buffer=None):
        super().__init__(name="Camera-Worker")
        self.cmd_queue = cmd_queue       # To receive high-level commands
        self.config = config
        self.serial_number = serial_number
        self.obs_buffer = obs_buffer     # Shared memory ring buffer for observations
        self.is_recording = False
        self.loop = True
        self.i=0

    def run(self):
        try:

            w, h = self.config["resolution"]
            fps = self.config["capture_fps"]
            align = rs.align(rs.stream.color)
            rs_config = rs.config()
            if self.config["enable_color"]:
                rs_config.enable_stream(rs.stream.color, 
                    w, h, rs.format.bgr8, fps)
            if self.config["enable_depth"]:
                rs_config.enable_stream(rs.stream.depth, 
                    w, h, rs.format.z16, fps)
            # if self.config["enable_infrared"]:
            #     rs_config.enable_stream(rs.stream.infrared, 1, 
            #         w, h, rs.format.rgb8, 30)

            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            print(f"[Camera-{self.serial_number}] Starting pipeline with config: {w}x{h}@{fps}fps")
            pipeline_profile = pipeline.start(rs_config)
            print(f"[Camera-{self.serial_number}] Pipeline started successfully {time.time():.3f}s")

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            device = pipeline_profile.get_device()
            sensors = device.query_sensors()
            for s in sensors:
                if s.is_color_sensor() and self.config["alignment"]:
                    s.set_option(rs.option.global_time_enabled, 1)

        
            while self.loop:
                now_epoch = time.time()
                
                # 1. Handle Commands
                self._handle_commands()

                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                if self.config["alignment"]:
                    frameset = align.process(frameset)
                depth = None
                color = None

                # realsense report in ms
                # data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000 - NOTE: we could convert capture timestamp to epoch timestamp
                if self.config["enable_color"]:
                    color_frame = frameset.get_color_frame()
                    color = np.asanyarray(color_frame.get_data())
                    # t = color_frame.get_timestamp() / 1000
                    # data['camera_capture_timestamp'] = t
                if self.config["enable_depth"]:
                    depth = np.asanyarray(
                        frameset.get_depth_frame().get_data())

                # Put observations in ring buffer (always, not just when recording)
                if self.obs_buffer is not None and color is not None:
                    obs_data = {
                        "color": color,
                        "timestamp": receive_time
                    }
                    try:
                        self.obs_buffer.put(obs_data, wait=False)
                    except Exception as e:
                        print(f"[Camera-Worker-{self.serial_number}] Failed to put obs in buffer: {e}")

                if color is not None:
                    window_name = f'RealSense-{self.serial_number}'
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(window_name, color)
                    cv2.waitKey(1)

                duration = time.time() - now_epoch
                frequency = np.round(1 / duration, 1)
                # print(f"FPS: {frequency}")

        except Exception as e:
            print(f"[Camera-Worker-{self.serial_number}] Critical Error: {e}")
        finally:
            print(f"[Camera-Worker-{self.serial_number}] Shutting down camera...")
            cv2.destroyAllWindows()
            pipeline.stop()

    def _handle_commands(self):
        if not self.cmd_queue.empty():
            cmd = self.cmd_queue.get()
            
            if cmd['type'] == 'EXIT': 
                self.loop = False
                cv2.destroyAllWindows()

            elif cmd['type'] == 'START_REC':
                self.is_recording = True
                self.episode_idx = cmd['ep_idx']

            elif cmd['type'] == 'STOP_REC':
                self.is_recording = False
                cv2.destroyAllWindows()

                # Send confirmation back to Conductor after saving
