import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

def download_and_setup_unrealcv():
    project_dir = Path(r"C:\Users\Long\Documents\Unreal Projects\FallSimulation")
    print(f"Dong thoi tim project khac trong thu muc Unreal Projects")
    if not project_dir.exists():
        # Thu tim cac ten project khac trong thu muc Unreal Projects
        unreal_projects_root = Path(r"C:\Users\Long\Documents\Unreal Projects")
        if unreal_projects_root.exists():
            projects = [p for p in unreal_projects_root.iterdir() if p.is_dir() and (p / "Config").exists()]
            if projects:
                project_dir = projects[0]
                print(f"Khong tim thay du an 'FallSimulation', tu dong chuyen sang: {project_dir}")
            else:
                print("Loi: Khong tim thay thu muc du an Unreal nao.")
                return
        else:
            print("Loi: Khong tim thay thu muc Unreal Projects.")
            return

    plugins_dir = project_dir / "Plugins"
    plugins_dir.mkdir(exist_ok=True)
    
    target_unrealcv_dir = plugins_dir / "unrealcv"
    if target_unrealcv_dir.exists():
        print(f"UnrealCV da ton tai o: {target_unrealcv_dir}. Dang tien hanh ghi de...")
        shutil.rmtree(target_unrealcv_dir)

    zip_url = "https://github.com/unrealcv/unrealcv/archive/refs/heads/5.2.zip"
    temp_zip = Path("unrealcv_temp.zip")
    extract_temp = Path("unrealcv_temp_extracted")

    print(f"Dang tai UnrealCV (nhanh 5.2) tu: {zip_url}...")
    try:
        urllib.request.urlretrieve(zip_url, temp_zip)
        print("Tai thanh cong. Dang giai nen...")
        
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
            
        # Thu muc goc giai nen se co ten 'unrealcv-5.2'
        extracted_root = extract_temp / "unrealcv-5.2"
        plugin_source = extracted_root / "plugins" / "unrealcv"
        
        if not plugin_source.exists():
            # Neu cau truc khac, thu tim xem
            plugin_source = extracted_root
            
        print(f"Dang cai dat plugin vao: {target_unrealcv_dir}...")
        shutil.copytree(plugin_source, target_unrealcv_dir)
        print("Cai dat thanh cong UnrealCV Plugin!")
        
    except Exception as e:
        print(f"Da xay ra loi trong qua trinh cai dat: {e}")
    finally:
        # Dọn dẹp file tạm
        if temp_zip.exists():
            temp_zip.unlink()
        if extract_temp.exists():
            shutil.rmtree(extract_temp)

if __name__ == "__main__":
    download_and_setup_unrealcv()
