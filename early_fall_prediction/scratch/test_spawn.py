import math
from unrealcv import Client
from PIL import Image
import io
from pathlib import Path

def test_spawn_camera():
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
    
    # Calculate position for Camera 1 (behind the player)
    distance = 250.0
    height = 100.0
    
    yaw_rad = math.radians(yaw)
    cx = px - distance * math.cos(yaw_rad)
    cy = py - distance * math.sin(yaw_rad)
    cz = pz + height
    
    # Move Camera 1 (index 1)
    client.request(f'vset /camera/1/location {cx} {cy} {cz}')
    client.request(f'vset /camera/1/rotation -15 {yaw} 0')
    
    print(f"Camera 1 moved to: {cx:.2f} {cy:.2f} {cz:.2f}")
    
    # Capture from Camera 1
    rgb_res = client.request('vget /camera/1/lit png')
    mask_res = client.request('vget /camera/1/object_mask png')
    
    if rgb_res and mask_res:
        rgb_img = Image.open(io.BytesIO(rgb_res))
        mask_img = Image.open(io.BytesIO(mask_res))
        
        rgb_img.save("data/ue5_raw/test_spawn_rgb.png")
        mask_img.save("data/ue5_raw/test_spawn_mask.png")
        print("Captured from Camera 1 successfully!")
    else:
        print("Error capturing from Camera 1.")

if __name__ == "__main__":
    test_spawn_camera()
