from unrealcv import Client
from PIL import Image
import io
from pathlib import Path

def test_match_player():
    client = Client(('127.0.0.1', 9000))
    client.connect()
    if not client.isconnected():
        print("Error: Could not connect to UE5.")
        return
        
    # Get active player camera manager transform
    loc_str = client.request('vget /object/PlayerCameraManager_0/location')
    rot_str = client.request('vget /object/PlayerCameraManager_0/rotation')
    
    print(f"Player Camera Manager Location: {loc_str}")
    print(f"Player Camera Manager Rotation: {rot_str}")
    
    px, py, pz = map(float, loc_str.split())
    pitch, yaw, roll = map(float, rot_str.split())
    
    # Sync Camera 0 to it
    client.request(f'vset /camera/0/location {px} {py} {pz}')
    client.request(f'vset /camera/0/rotation {pitch} {yaw} {roll}')
    
    # Capture RGB and Mask
    rgb_res = client.request('vget /camera/0/lit png')
    mask_res = client.request('vget /camera/0/object_mask png')
    
    if rgb_res and mask_res:
        rgb_img = Image.open(io.BytesIO(rgb_res))
        mask_img = Image.open(io.BytesIO(mask_res))
        
        rgb_img.save("data/ue5_raw/test_match_rgb.png")
        mask_img.save("data/ue5_raw/test_match_mask.png")
        print("Successfully matched and captured active player view!")
    else:
        print("Error capturing matched view.")

if __name__ == "__main__":
    test_match_player()
