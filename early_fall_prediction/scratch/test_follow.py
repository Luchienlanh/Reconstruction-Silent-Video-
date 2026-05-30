import math
from unrealcv import Client
from PIL import Image
import io
from pathlib import Path

def test_camera_follow():
    client = Client(('127.0.0.1', 9000))
    client.connect()
    if not client.isconnected():
        print("Error: Could not connect to UE5.")
        return
        
    # 1. Lay toa do nguoi choi
    loc_str = client.request('vget /object/BP_ThirdPersonCharacter_C_0/location')
    rot_str = client.request('vget /object/BP_ThirdPersonCharacter_C_0/rotation')
    
    print(f"Player Location: {loc_str}")
    print(f"Player Rotation: {rot_str}")
    
    px, py, pz = map(float, loc_str.split())
    pitch, yaw, roll = map(float, rot_str.split())
    
    # 2. Tinh toan toa do camera (dung phia sau nguoi choi 250cm, cao hon 120cm)
    distance = 250.0
    height = 120.0
    
    yaw_rad = math.radians(yaw)
    cx = px - distance * math.cos(yaw_rad)
    cy = py - distance * math.sin(yaw_rad)
    cz = pz + height
    
    # 3. Di chuyen camera va quay camera huong ve nguoi choi (pitch = -15 de chui xuong)
    client.request(f'vset /camera/0/location {cx} {cy} {cz}')
    client.request(f'vset /camera/0/rotation -15 {yaw} 0')
    
    print(f"Camera moved to: {cx:.2f} {cy:.2f} {cz:.2f}")
    
    # 4. Chup anh thu nghiem
    rgb_res = client.request('vget /camera/0/lit png')
    if rgb_res and isinstance(rgb_res, bytes):
        img = Image.open(io.BytesIO(rgb_res))
        out_path = Path("data/ue5_raw/test_follow.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"Chup thanh cong! Anh duoc luu tai: {out_path.absolute()}")
    else:
        print("Loi chup anh.")

if __name__ == "__main__":
    test_camera_follow()
