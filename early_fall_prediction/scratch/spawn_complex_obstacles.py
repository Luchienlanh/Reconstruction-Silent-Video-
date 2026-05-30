"""
spawn_complex_obstacles.py
Sinh dia hinh phuc tap voi 4 loai vat can trong Unreal Engine 5.

FIXED:
  - Tinh Z chinh xac: lay Z mat san tu player (feet) thay vi body center
  - Spawn mot so vat the o TRAM NHIN GAO (ngay truoc mat nhan vat)
  - Vat can cao hon, ro rang hon trong viewport

Class -> Mau sac phan vung (Segmentation Mask):
  Class 1 - obstacle   : hop cao / vat can     -> Mau Do     RGB(255,   0,   0)
  Class 2 - pothole    : o ga / ho sau         -> Mau Xanh D RGB(  0,   0, 255)
  Class 3 - step_edge  : bac thang / go bac    -> Mau Xanh L RGB(  0, 255,   0)
  Class 4 - wet_surface: mat duong tron/uot    -> Mau Vang   RGB(255, 255,   0)
"""

import random
import math
import time
from unrealcv import Client


# ------------------------------------------------------------------ #
#  CAU HINH                                                            #
# ------------------------------------------------------------------ #
UE5_HOST  = ('127.0.0.1', 9000)

# Pham vi diem spawn (UE5 ThirdPerson template, tinh bang cm)
ARENA_X   = (-650, 650)
ARENA_Y   = (-1050, 1050)
SAFE_DIST = 250.0   # Khoang cach toi thieu tu player de khong spawn chinh len nguoi (cm)

# Khoang cach "vung near" – spawn mot so vat the o day de nhan vat co the nhin thay ngay
NEAR_MIN  = 300.0
NEAR_MAX  = 500.0

# Loai vat can
TERRAIN_TYPES = {
    "obstacle": {
        "color":  (255, 0,   0  ),
        "scale_x": (0.8, 2.0),
        "scale_y": (0.8, 2.0),
        "scale_z": (1.5, 3.5),   # Vat can cao, de thay
        "z_base":  "ground",     # Dat tren mat dat
        "z_extra": 5,            # Tong offset (cm)
    },
    "pothole": {
        "color":  (0,   0,   255),
        "scale_x": (1.0, 2.5),
        "scale_y": (1.0, 2.5),
        "scale_z": (0.08, 0.18), # Rat mong, sat san
        "z_base":  "ground",
        "z_extra": -8,           # Chim nhe xuong mat san
    },
    "step_edge": {
        "color":  (0,   255, 0  ),
        "scale_x": (1.5, 3.5),   # Dai theo mot chieu, tao go
        "scale_y": (0.3, 0.8),
        "scale_z": (0.2, 0.5),   # Cao vua phai (gờ bậc)
        "z_base":  "ground",
        "z_extra": 0,
    },
    "wet_surface": {
        "color":  (255, 255, 0  ),
        "scale_x": (2.0, 4.0),
        "scale_y": (2.0, 4.0),
        "scale_z": (0.03, 0.07), # Cuc mong – vung nuoc
        "z_base":  "ground",
        "z_extra": 3,            # Noi nhe len mat san de camera nhin thay
    },
}

# So luong vat the tung loai
COUNTS = {"obstacle": 10, "pothole": 8, "step_edge": 8, "wet_surface": 6}

# Trong moi loai, bao nhieu vat the spawn trong vung "near" (nhin thay ngay)
NEAR_COUNTS = {"obstacle": 3, "pothole": 2, "step_edge": 2, "wet_surface": 2}

DELAY_REQ = 0.07    # Giay giua moi lenh UnrealCV
DELAY_OBJ = 0.12    # Giay giua moi vat the


# ------------------------------------------------------------------ #
#  HELPER                                                              #
# ------------------------------------------------------------------ #
def safe_req(client, cmd):
    time.sleep(DELAY_REQ)
    try:
        return client.request(cmd)
    except Exception as e:
        print(f"[WARN] {cmd[:55]}... => {e}")
        return None


def get_player_transform(client):
    loc = safe_req(client, 'vget /object/BP_ThirdPersonCharacter_C_0/location')
    rot = safe_req(client, 'vget /object/BP_ThirdPersonCharacter_C_0/rotation')
    px, py, pz, yaw = 0.0, 0.0, 0.0, 0.0
    if loc and "error" not in str(loc).lower():
        try:
            px, py, pz = map(float, loc.split())
        except Exception:
            pass
    if rot and "error" not in str(rot).lower():
        try:
            _, yaw, _ = map(float, rot.split())
        except Exception:
            pass
    # Trong UE5 ThirdPerson, origin cua character la o chan (Capsule base)
    # nen pz xap xi = mat dat
    ground_z = pz
    return px, py, ground_z, yaw


def pos_near_player(px, py, yaw, min_dist, max_dist):
    """Chon vi tri ngau nhien trong vung truoc mat nhan vat."""
    dist  = random.uniform(min_dist, max_dist)
    angle = math.radians(yaw) + random.uniform(-math.pi / 2, math.pi / 2)
    rx    = px + dist * math.cos(angle)
    ry    = py + dist * math.sin(angle)
    return rx, ry


def pos_random_arena(px, py):
    """Chon vi tri ngau nhien trong toan bo arena."""
    for _ in range(80):
        rx = random.uniform(*ARENA_X)
        ry = random.uniform(*ARENA_Y)
        if math.hypot(rx - px, ry - py) > SAFE_DIST:
            return rx, ry
    return random.uniform(*ARENA_X), random.uniform(*ARENA_Y)


def cleanup(client):
    print("[INFO] Don dep vat the cu (test_cube_*)...")
    res = safe_req(client, 'vget /objects')
    if not res:
        return
    removed = 0
    for obj in res.split():
        if "test_cube_" in obj:
            safe_req(client, f'vset /object/{obj}/destroy')
            removed += 1
    print(f"[INFO] Da xoa {removed} vat the cu.")


# ------------------------------------------------------------------ #
#  SPAWN CHINH                                                         #
# ------------------------------------------------------------------ #
def spawn_complex_terrain():
    print("=" * 64)
    print("SPAWN COMPLEX TERRAIN  –  4 loai dia hinh nguy hiem")
    print("=" * 64)

    client = Client(UE5_HOST)
    client.connect()
    if not client.isconnected():
        print("[ERROR] Khong the ket noi voi UE5 (port 9000).")
        print("         Hay chac chan UE5 dang chay Play-in-Editor.")
        return

    print("[OK] Ket noi voi Unreal Engine 5 thanh cong.")

    px, py, gz, yaw = get_player_transform(client)
    print(f"[INFO] Vi tri nhan vat : X={px:.0f} Y={py:.0f}")
    print(f"[INFO] Z mat dat (uoc tinh): {gz:.0f}  |  Yaw: {yaw:.1f} deg")

    cleanup(client)

    total_spawned = 0
    obj_index     = 0

    for terrain_label, cfg in TERRAIN_TYPES.items():
        count_total = COUNTS[terrain_label]
        count_near  = NEAR_COUNTS[terrain_label]
        color       = cfg["color"]
        r, g, b     = color

        print(f"\n[SPAWN] {terrain_label.upper()} x{count_total} | color=RGB{color}")

        spawned = 0
        for k in range(count_total):
            obj_name = f"test_cube_{obj_index}"
            obj_index += 1

            # Chon vi tri: mot so gan, con lai toan bo arena
            if k < count_near:
                rx, ry = pos_near_player(px, py, yaw, NEAR_MIN, NEAR_MAX)
            else:
                rx, ry = pos_random_arena(px, py)

            # Z = mat dat + offset tung loai
            rz = gz + cfg["z_extra"]

            # Spawn cube
            res = safe_req(client, f'vset /objects/spawn_cube {obj_name}')
            if res is None or "error" in str(res).lower():
                print(f"  [FAIL] {obj_name}: {res}")
                continue

            time.sleep(DELAY_OBJ)

            # Scale
            sx = random.uniform(*cfg["scale_x"])
            sy = random.uniform(*cfg["scale_y"])
            sz = random.uniform(*cfg["scale_z"])
            safe_req(client, f'vset /object/{obj_name}/scale {sx:.3f} {sy:.3f} {sz:.3f}')

            # Vi tri
            safe_req(client, f'vset /object/{obj_name}/location {rx:.1f} {ry:.1f} {rz:.1f}')

            # Mau sac phan vung
            safe_req(client, f'vset /object/{obj_name}/color {r} {g} {b}')

            spawned      += 1
            total_spawned += 1

            near_tag = " [NEAR-PLAYER]" if k < count_near else ""
            print(f"  [{spawned}/{count_total}] {obj_name}: "
                  f"pos=({rx:.0f},{ry:.0f},{rz:.0f}) "
                  f"scale=({sx:.2f},{sy:.2f},{sz:.2f})"
                  f"{near_tag}")

        print(f"  -> Da spawn {spawned}/{count_total} {terrain_label}")

    # To den nhan vat – tranh nham la vat can do
    safe_req(client, 'vset /object/BP_ThirdPersonCharacter_C_0/color 0 0 0')

    print("\n" + "=" * 64)
    print(f"[DONE] Tong so vat the da spawn: {total_spawned}")
    print()
    print("  MA SAC PHAN VUNG (Segmentation Mask):")
    print("    Mau Do         RGB(255,   0,   0) -> class 1: obstacle")
    print("    Mau Xanh Duong RGB(  0,   0, 255) -> class 2: pothole")
    print("    Mau Xanh La    RGB(  0, 255,   0) -> class 3: step_edge")
    print("    Mau Vang       RGB(255, 255,   0) -> class 4: wet_surface")
    print()
    print("  LUU Y:")
    print("    - Mot so vat can da duoc spawn TRUOC MAT nhan vat (NEAR-PLAYER)")
    print("    - Nhin xung quanh trong viewport UE5 de thay cac mau khac nhau")
    print("    - Potholes & wet_surfaces rat mong, nhin gan mat dat se ro hon")
    print()
    print("  BUOC TIEP THEO:")
    print("    1. Di chuyen nhan vat qua cac vat can trong UE5")
    print("    2. Chay: python training/ue5_capture.py --num-frames 300")
    print("=" * 64)


if __name__ == "__main__":
    spawn_complex_terrain()
