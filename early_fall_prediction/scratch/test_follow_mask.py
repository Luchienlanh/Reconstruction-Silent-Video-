import math
from unrealcv import Client
from PIL import Image
import io
import numpy as np
from pathlib import Path

def test_camera_follow_mask():
    client = Client(('127.0.0.1', 9000))
    client.connect()
    if not client.isconnected():
        print("Error: Could not connect to UE5.")
        return
        
    # Get player transform
    loc_str = client.request('vget /object/BP_ThirdPersonCharacter_C_0/location')
    rot_str = client.request('vget /object/BP_ThirdPersonCharacter_C_0/rotation')
    
    px, py, pz = map(float, loc_str.split())
    pitch, yaw, roll = map(float, rot_str.split())
    
    print(f"Player: Pos=({px}, {py}, {pz}), Rot=({pitch}, {yaw}, {roll})")
    
    # Place camera behind player
    distance = 200.0 # Closer
    height = 80.0
    
    yaw_rad = math.radians(yaw)
    cx = px - distance * math.cos(yaw_rad)
    cy = py - distance * math.sin(yaw_rad)
    cz = pz + height
    
    client.request(f'vset /camera/0/location {cx} {cy} {cz}')
    # Pitch -10 to look slightly down
    client.request(f'vset /camera/0/rotation -10 {yaw} 0')
    
    # Capture RGB and Mask
    rgb_res = client.request('vget /camera/0/lit png')
    mask_res = client.request('vget /camera/0/object_mask png')
    
    if rgb_res and mask_res:
        rgb_img = Image.open(io.BytesIO(rgb_res))
        mask_img = Image.open(io.BytesIO(mask_res))
        
        rgb_img.save("data/ue5_raw/test_follow_rgb.png")
        mask_img.save("data/ue5_raw/test_follow_mask.png")
        
        # Analyze mask colors
        mask_arr = np.array(mask_img)
        # Unique colors and their counts
        colors, counts = np.unique(mask_arr.reshape(-1, 3), axis=0, return_counts=True)
        
        print("\nColors found in mask:")
        char_found = False
        for c, cnt in zip(colors, counts):
            print(f"Color {c}: {cnt} pixels")
            # Character color is R=0, G=191, B=0. PIL images are RGB.
            if c[0] == 0 and c[1] == 191 and c[2] == 0:
                print(f"--> FOUND character! {cnt} pixels")
                char_found = True
                
        if not char_found:
            print("--> Character NOT found in the mask.")
    else:
        print("Error capturing images.")

if __name__ == "__main__":
    test_camera_follow_mask()
