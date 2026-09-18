import argparse
from pathlib import Path
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Verify submission folder before upload")
    parser.add_argument("--denoised_dir", type=str, required=True, 
                        help="Path to the folder containing the final 20 denoised images")
    return parser.parse_args()

def main():
    args = parse_args()
    denoised_dir = Path(args.denoised_dir)
    
    if not denoised_dir.exists():
        print(f"ERROR: Directory '{denoised_dir}' does not exist.")
        return

    # The preliminary submission requires exactly images 461 to 480
    expected_files = [f"{i}.png" for i in range(461, 481)]
    
    missing_files = []
    invalid_files = []
    
    print(f"Verifying images in: {denoised_dir}")
    print("-" * 40)
    
    for fname in expected_files:
        path = denoised_dir / fname
        
        if not path.exists():
            missing_files.append(fname)
            continue
            
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                invalid_files.append(f"{fname} (Expected RGB, got {img.mode})")
            if img.size != (992, 992):
                invalid_files.append(f"{fname} (Expected 992x992, got {img.size})")
            print(f"[OK] {fname} - {img.size}, {img.mode}")
        except Exception as e:
            invalid_files.append(f"{fname} (Failed to read: {str(e)})")

    print("-" * 40)
    if missing_files:
        print(f"[FAIL] Missing {len(missing_files)} expected files:")
        for m in missing_files:
            print(f"  - {m}")
    elif invalid_files:
        print(f"[FAIL] Found {len(invalid_files)} invalid files:")
        for inv in invalid_files:
            print(f"  - {inv}")
    else:
        print(f"[SUCCESS] All {len(expected_files)} files verified and ready for submission!")

if __name__ == "__main__":
    main()
