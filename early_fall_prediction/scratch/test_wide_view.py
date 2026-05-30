import math
from unrealcv import Client
from PIL import Image
import io
from pathlib import Path

def test_wide_view():
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
    
    # Place camera far away and high up
    distance = 600.0 
    height = 400.0
    
    yaw_rad = math.radians(yaw)
    cx = px - distance * math.cos(yaw_rad)
    cy = py - distance * math.sin(yaw_rad)
    cz = pz + height
    
    # Move camera
    client.request(f'vset /camera/0/location {cx} {cy} {cz}')
    # Pitch -35 to look down
    client.request(f'vset /camera/0/rotation -35 {yaw} 0')
    
    # Capture RGB and Mask
    rgb_res = client.request('vget /camera/0/lit png')
    mask_res = client.request('vget /camera/0/object_mask png')
    
    if rgb_res and mask_res:
        rgb_img = Image.open(io.BytesIO(rgb_res))
        mask_img = Image.open(io.BytesIO(mask_res))
        
        rgb_img.save("data/ue5_raw/test_wide_rgb.png")
        mask_img.save("data/ue5_raw/test_wide_mask.png")
        print("Wide view captured successfully!")
    else:
        print("Error capturing wide view.")

if __name__ == "__main__":
    test_wide_view()
