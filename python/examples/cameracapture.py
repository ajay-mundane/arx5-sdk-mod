from collections import defaultdict
import multiprocessing as mp
import time
import numpy as np
import pickle
import sys
import os
import pyrealsense2 as rs


# --- The Camera Process Class ---
class CameraCaptureProcess(mp.Process):
    def __init__(self, cmd_queue, data_queue, config, serial_number):
        super().__init__(name="Robot-Worker")
        self.cmd_queue = cmd_queue       # To receive high-level commands
        self.data_queue = data_queue   # To send latest state for UI/Monitoring
        self.config = config
        self.serial_number = serial_number
        self.is_recording = False
        self.loop = True
        self.image_buffer = {"timestamps":[], 'color':np.zeros((300,480,640,3)), 'depth':np.zeros((300,480,640))}
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
            # if self.enable_infrared:
            #     rs_config.enable_stream(rs.stream.infrared,
            #         w, h, rs.format.y8, fps)

            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            device = pipeline_profile.get_device()
            sensors = device.query_sensors()
            for s in sensors:
                if s.is_color_sensor():
                    s.set_option(rs.option.global_time_enabled, 1)

        
            while self.loop:
                now_epoch = time.time()
                
                # 1. Handle Commands
                self._handle_commands()

                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                # frameset = align.process(frameset)

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

                # print(depth.shape)

                if self.is_recording:
                    print("PUTTING DATA RIGHT NOW")
                    self._put_data(receive_time, color, depth)


                duration = time.time() - now_epoch
                frequency = np.round(1 / duration, 1)
                # print(f"FPS: {frequency}")

        except Exception as e:
            print(f"[Camera-Worker-{self.serial_number}] Critical Error: {e}")
        finally:
            print(f"[Camera-Worker-{self.serial_number}] Shutting down camera...")
            pipeline.stop()

    def _handle_commands(self):
        if not self.cmd_queue.empty():
            cmd = self.cmd_queue.get()
            
            if cmd['type'] == 'EXIT': self.loop = False

            elif cmd['type'] == 'START_REC':
                self.is_recording = True
                self.episode_idx = cmd['ep_idx']

            elif cmd['type'] == 'STOP_REC':
                self.is_recording = False
                self._send_data()
                # Send confirmation back to Conductor after saving
                

    def _put_data(self, receive_time, color, depth):
        self.image_buffer['timestamps'].append(receive_time)
        self.image_buffer["color"][self.i] = color
        self.image_buffer['depth'][self.i] = depth
        self.i += 1
    
    def _send_data(self):
        for key, val in self.image_buffer.items():
            self.image_buffer[key] = np.array(val)
        self.data_queue.put(dict(self.image_buffer))
        self.image_buffer.clear()
