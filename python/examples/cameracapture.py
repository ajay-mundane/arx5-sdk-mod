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
    def __init__(self, cmd_queue, data_queue, config, serial_number):
        super().__init__(name="Robot-Worker")
        self.cmd_queue = cmd_queue       # To receive high-level commands
        self.data_queue = data_queue   # To send latest state for UI/Monitoring
        self.config = config
        self.serial_number = serial_number
        self.is_recording = False
        self.loop = True
        self.image_buffer = {"cam_timestamps":NumpyAccumulator(dtype=np.float64), 'color': NumpyAccumulator(shape_suffix=(480,640,3),dtype=np.uint8), 'depth': NumpyAccumulator(shape_suffix=(480,640), dtype=np.uint16)}
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

                # print(depth.shape)

                if self.is_recording:
                    self._put_data(receive_time, color, depth)
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
            
            if cmd['type'] == 'EXIT': self.loop = False

            elif cmd['type'] == 'START_REC':
                self.is_recording = True
                self.episode_idx = cmd['ep_idx']

            elif cmd['type'] == 'STOP_REC':
                self.is_recording = False
                self._send_data()
                cv2.destroyAllWindows()

                # Send confirmation back to Conductor after saving
                

    def _put_data(self, receive_time, color, depth):
        self.image_buffer['cam_timestamps'].append(receive_time)
        if color is not None:
            self.image_buffer["color"].append(color)
        if depth is not None:
            self.image_buffer['depth'].append(depth)
    
    def _send_data(self):
        # Send data in chunks to avoid memory issues with large recordings
        CHUNK_SIZE = 200  # frames per chunk
        
        # Get data arrays
        data = {k: v.data for k, v in self.image_buffer.items()}
        total_frames = len(data['cam_timestamps'])
        
        if total_frames == 0:
            # Send empty data indicator
            self.data_queue.put({'chunk_id': 0, 'total_chunks': 1, 'data': data})
        else:
            # Calculate number of chunks needed
            total_chunks = (total_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            # Send data in chunks with progress bar
            with tqdm(total=total_chunks, desc=f"Camera-{self.serial_number} sending", unit="chunk") as pbar:
                for chunk_id in range(total_chunks):
                    start_idx = chunk_id * CHUNK_SIZE
                    end_idx = min(start_idx + CHUNK_SIZE, total_frames)
                    
                    chunk_data = {}
                    for key, value in data.items():
                        chunk_data[key] = value[start_idx:end_idx]
                    
                    chunk_msg = {
                        'chunk_id': chunk_id,
                        'total_chunks': total_chunks,
                        'data': chunk_data
                    }
                    
                    self.data_queue.put(chunk_msg)
                    pbar.update(1)
        
        # Reset buffers
        for k, v in self.image_buffer.items():
            v.reset()
