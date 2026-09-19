import argparse
import zipfile
from pathlib import Path
from PIL import Image
import io

def parse_args():
    parser = argparse.ArgumentParser(description="Verify submission ZIP before upload")
    parser.add_argument("--zip_path", type=str, required=True, 
                        help="Path to the TeamName.zip file")
    return parser.parse_args()

def verify_submission(zip_path):
    errors = []
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        print(f"ERROR: ZIP file '{zip_path}' does not exist.")
        return False
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            files = z.namelist()
            expected = [f"{i}.png" for i in range(461, 481)]
            
            if len(files) != 20:
                errors.append(f"Expected exactly 20 files, found {len(files)}")
            
            for fname in expected:
                if fname not in files:
                    errors.append(f"Missing required file: {fname}")
                else:
                    with z.open(fname) as f:
                        try:
                            # Use io.BytesIO to read from zip into PIL
                            img_data = io.BytesIO(f.read())
                            img = Image.open(img_data)
                            if img.mode != "RGB":
                                errors.append(f"{fname}: mode={img.mode}, expected RGB")
                            if img.size != (992, 992):
                                errors.append(f"{fname}: size={img.size}, expected (992, 992)")
                        except Exception as e:
                            errors.append(f"{fname}: Failed to open image - {str(e)}")
            
            for f in files:
                if f not in expected:
                    errors.append(f"Unexpected extra file found: {f}")
    except zipfile.BadZipFile:
        errors.append(f"File '{zip_path}' is not a valid ZIP archive.")
        
    if errors:
        print("❌ SUBMISSION VERIFICATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✅ Submission verified successfully! Everything looks perfect.")
        return True

def main():
    args = parse_args()
    verify_submission(args.zip_path)

if __name__ == "__main__":
    main()
