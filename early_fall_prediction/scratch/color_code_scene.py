from unrealcv import Client
import time

def color_code_scene():
    client = Client(('127.0.0.1', 9000))
    client.connect()
    if not client.isconnected():
        print("Error: Could not connect to UE5.")
        return
        
    print("Querying all objects in the scene...")
    objects = client.request('vget /objects').split()
    print(f"Found {len(objects)} total objects.")
    
    red_count = 0
    black_count = 0
    
    for idx, obj in enumerate(objects):
        if "StaticMeshActor" in obj:
            # Get material of the object
            mat = client.request(f'vget /object/{obj}/material').strip()
            
            # 1. Ramps and obstacles (MI_ThirdPersonColWay) -> set to pure Red (R=255, G=0, B=0)
            if "MI_ThirdPersonColWay" in mat:
                client.request(f'vset /object/{obj}/color 255 0 0')
                red_count += 1
            # 2. Floor grid (MI_PrototypeGrid_Gray) -> set to pure Black (R=0, G=0, B=0) to ignore
            elif "MI_PrototypeGrid_Gray" in mat:
                client.request(f'vset /object/{obj}/color 0 0 0')
                black_count += 1
                
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(objects)} objects...")
            
    print(f"\nColor coding completed!")
    print(f"-> Set {red_count} actors (ramps/obstacles) to RED (255, 0, 0).")
    print(f"-> Set {black_count} actors (floor/safe ground) to BLACK (0, 0, 0).")
    
    # Hide the player character from the segmentation mask or set its color to Black so it doesn't interfere
    client.request('vset /object/BP_ThirdPersonCharacter_C_0/color 0 0 0')
    print("-> Set player character to BLACK (0, 0, 0) to avoid false hazard labels.")

if __name__ == "__main__":
    color_code_scene()
