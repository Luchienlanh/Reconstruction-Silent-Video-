import os
import time
import io
import json
import math
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import argparse

try:
    from unrealcv import Client
except ImportError:
    print("Warning: unrealcv is not installed. Run: pip install unrealcv")
    Client = None


class UE5DataCapture:
    """Capture RGB, Depth, and Segmentation Mask from Unreal Engine 5."""
    
    def __init__(self, output_dir="data/ue5_raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.camera_id = 0
        
        # Connect to UE5
        if Client:
            self.client = Client(('127.0.0.1', 9000))
            self.client.connect()
            if not self.client.isconnected():
                print("Error: Cannot connect to Unreal Engine. Make sure UE5 is running and UnrealCV is active.")
            else:
                print("Successfully connected to Unreal Engine!")
                
                # Check cameras and spawn a new free-moving camera if needed
                cameras = self.client.request('vget /cameras').split()
                print(f"Current cameras in scene: {cameras}")
                if len(cameras) < 2:
                    print("Spawning a new free-moving camera actor...")
                    self.client.request('vset /cameras/spawn')
                    cameras = self.client.request('vget /cameras').split()
                    print(f"Updated cameras in scene: {cameras}")
                
                # Camera index 1 corresponds to the spawned FusionCameraSensor
                self.camera_id = 1
                print(f"Using camera sensor index {self.camera_id} for third-person capture.")
        else:
            print("Error: unrealcv module not installed.")
            
    def set_camera(self, loc, rot):
        """Di chuyen camera toi vi tri va goc quay chi dinh
        loc: [x, y, z]
        rot: [pitch, yaw, roll]
        """
        if not self.client or not self.client.isconnected():
            return
            
        loc_str = f"{loc[0]} {loc[1]} {loc[2]}"
        rot_str = f"{rot[0]} {rot[1]} {rot[2]}"
        
        self.client.request(f'vset /camera/{self.camera_id}/location {loc_str}')
        self.client.request(f'vset /camera/{self.camera_id}/rotation {rot_str}')
        
    def capture_frame(self, frame_id: int):
        """Chup 1 frame gom: RGB + Depth + Object Mask"""
        if not self.client or not self.client.isconnected():
            print(f"Skipping capture_frame {frame_id} - no connection.")
            return
            
        print(f"Capturing Frame {frame_id:05d}...")
        
        # 1. RGB Capture
        rgb_res = self.client.request(f'vget /camera/{self.camera_id}/lit png')
        if rgb_res and isinstance(rgb_res, bytes):
            rgb_img = Image.open(io.BytesIO(rgb_res))
            rgb_img.save(self.output_dir / f"rgb_{frame_id:05d}.png")
            
        # 2. Depth Map Capture
        depth_res = self.client.request(f'vget /camera/{self.camera_id}/depth npy')
        if depth_res and isinstance(depth_res, bytes):
            depth_arr = np.load(io.BytesIO(depth_res))
            np.save(self.output_dir / f"depth_{frame_id:05d}.npy", depth_arr)
            
        # 3. Object Mask Capture (Segmentation)
        mask_res = self.client.request(f'vget /camera/{self.camera_id}/object_mask png')
        if mask_res and isinstance(mask_res, bytes):
            mask_img = Image.open(io.BytesIO(mask_res))
            mask_img.save(self.output_dir / f"mask_{frame_id:05d}.png")
            
        # Ghi nhan thoi diem chup
        with open(self.output_dir / "timestamps.txt", "a") as f:
            f.write(f"{frame_id},{time.time()}\n")

    def list_objects(self):
        """Liet ke cac object co trong scene de gan mau segmentation."""
        if not self.client or not self.client.isconnected():
            return []
        res = self.client.request('vget /objects')
        if res:
            objects = res.split(' ')
            return objects
        return []
        
    def set_object_color(self, object_name, r, g, b):
        """Gan mau cho object de xuat ra Segmentation Mask."""
        if not self.client or not self.client.isconnected():
            return
        self.client.request(f'vset /object/{object_name}/color {r} {g} {b}')

def run_simulation_capture(num_frames=100, output_dir="data/ue5_raw"):
    capture = UE5DataCapture(output_dir=output_dir)
    if not capture.client or not capture.client.isconnected():
        print("Error: UnrealCV client is not connected. Exiting.")
        return
        
    print(f"Starting Unreal Engine 5 simulation data capture ({num_frames} frames)...")
    
    # Xoa file timestamp cu
    ts_file = Path(output_dir) / "timestamps.txt"
    if ts_file.exists():
        ts_file.unlink()
        
    for i in range(num_frames):
        # Query player transform dynamically on each frame
        loc_str = capture.client.request('vget /object/BP_ThirdPersonCharacter_C_0/location')
        rot_str = capture.client.request('vget /object/BP_ThirdPersonCharacter_C_0/rotation')
        
        if loc_str and rot_str and "error" not in loc_str:
            px, py, pz = map(float, loc_str.split())
            pitch, yaw, roll = map(float, rot_str.split())
            
            # Position Camera 1 (index 1) exactly 250cm behind the player, 100cm higher
            distance = 250.0
            height = 100.0
            yaw_rad = math.radians(yaw)
            cx = px - distance * math.cos(yaw_rad)
            cy = py - distance * math.sin(yaw_rad)
            cz = pz + height
            
            # Orient Camera 1 to look down slightly (-15 pitch) and match character yaw
            capture.set_camera([cx, cy, cz], [-15, yaw, 0])
        else:
            print(f"Warning: Could not get player location at frame {i:05d}")
            
        # Capture the frame
        capture.capture_frame(i)
        time.sleep(0.05) # Wait for engine to render
        
    print(f"Completed! Data saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unreal Engine 5 Data Capture Script")
    parser.add_argument("--num-frames", type=int, default=10, help="So luong frame can chup")
    parser.add_argument("--out-dir", type=str, default="data/ue5_raw", help="Thu muc luu")
    
    args = parser.parse_args()
    
    run_simulation_capture(num_frames=args.num_frames, output_dir=args.out_dir)
