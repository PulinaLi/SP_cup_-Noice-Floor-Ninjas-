# 08 — Submission Checklist

## 8.1 Preliminary Round Deliverables

### Google Drive Folder Contents

| # | Item | Format | Contents |
|---|------|--------|----------|
| 1 | `TeamName.zip` | ZIP file | Exactly 20 denoised PNGs: `461.png` – `480.png` |
| 2 | `TeamName_Report.pdf` | PDF | Solution report (max 5 pages) |

### GitHub Repository

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Private GitHub repo created | ☐ |
| 2 | Organizer account added as collaborator | ☐ |
| 3 | Preserves starter-repository structure | ☐ |
| 4 | All custom code in `scripts/` only | ☐ |
| 5 | `scripts/denoise.py` present and working | ☐ |
| 6 | `scripts/README.md` with clear instructions | ☐ |
| 7 | `scripts/requirements.txt` with dependencies | ☐ |
| 8 | Git commit SHA recorded in README | ☐ |
| 9 | No post-deadline commits | ☐ |

### Model/Checkpoint (if applicable)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Model uploaded to Google Drive | ☐ |
| 2 | Link accessible to organizers | ☐ |
| 3 | Filename documented in `scripts/README.md` | ☐ |
| 4 | Expected local path documented | ☐ |
| 5 | SHA-256 checksum documented | ☐ |
| 6 | Model file frozen (no changes after deadline) | ☐ |

---

## 8.2 Image Submission Checklist

### Denoised Output Verification

```
☐ Exactly 20 files in the ZIP
☐ Filenames: 461.png, 462.png, ..., 480.png
☐ All files are PNG format
☐ All images are RGB (3 channels)
☐ All images are 992 × 992 pixels
☐ No _noise suffix in output filenames
☐ No extra files in the ZIP
☐ Images look visually correct (spot-check)
```

### Automated Verification Script

```python
"""Run this before submission to verify everything."""
import zipfile
from pathlib import Path
from PIL import Image

def verify_submission(zip_path):
    errors = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = z.namelist()
        expected = [f"{i}.png" for i in range(461, 481)]
        
        # Check file count
        if len(files) != 20:
            errors.append(f"Expected 20 files, found {len(files)}")
        
        # Check each file
        for fname in expected:
            if fname not in files:
                errors.append(f"Missing: {fname}")
            else:
                # Extract and verify
                with z.open(fname) as f:
                    img = Image.open(f)
                    if img.mode != "RGB":
                        errors.append(f"{fname}: mode={img.mode}, expected RGB")
                    if img.size != (992, 992):
                        errors.append(f"{fname}: size={img.size}, expected (992, 992)")
        
        # Check for extra files
        for f in files:
            if f not in expected:
                errors.append(f"Unexpected file: {f}")
    
    if errors:
        print("❌ SUBMISSION VERIFICATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("✅ Submission verified successfully!")
        return True

# Usage:
# verify_submission("TeamName.zip")
```

---

## 8.3 Code Verification Checklist

### Before Submission

```
☐ scripts/denoise.py runs successfully
☐ The denoise script handles --noise_dir and --denoised_dir arguments
☐ Running denoise.py produces the exact submitted images
☐ Code runs on CPU (no CUDA-only code paths)
☐ Code works offline (no network requests)
☐ All dependencies listed in requirements.txt
☐ No hardcoded image-specific logic
☐ No modifications to baseline/ or evaluation/ directories
☐ README.md has complete setup instructions
☐ Code is clean and readable
```

### Verification Commands

```bash
# 1. Fresh environment test
pip install -r scripts/requirements.txt

# 2. Run denoising on submission images (CPU)
python scripts/denoise.py \
    --noise_dir competition_data/submissions/noisy \
    --denoised_dir competition_data/submissions/denoised

# 3. Verify outputs match submitted images
# (Compare byte-for-byte with the images in your ZIP)

# 4. Run on public set for score check
python scripts/denoise.py \
    --noise_dir competition_data/public/noisy \
    --denoised_dir competition_data/public/denoised

python evaluation/evaluate.py \
    --noisy_dir competition_data/public/noisy \
    --pred_dir competition_data/public/denoised \
    --gt_dir competition_data/public/ground_truth

# 5. Compute model checksum
python -c "
import hashlib
with open('scripts/checkpoints/best_model.pth', 'rb') as f:
    sha256 = hashlib.sha256(f.read()).hexdigest()
print(f'SHA-256: {sha256}')
"

# 6. Record git commit SHA
git log -1 --format='%H'
```

---

## 8.4 Report Compliance Checklist

```
☐ Max 5 pages
☐ 12-point font
☐ 1-inch margins on all sides
☐ Filename: TeamName_Report.pdf
☐ No cover page required (but can include)
☐ Includes:
    ☐ Brief introduction to the problem
    ☐ Problem observations and dataset analysis
    ☐ Solution architecture (with diagrams)
    ☐ Justification of selected approach
    ☐ Alternative approaches considered
    ☐ Methodology for building the algorithm/model
    ☐ Validation approach description
    ☐ Results with metrics
☐ Clear flow and proper justification
☐ Classical SP techniques documented (for 10% bonus)
```

---

## 8.5 Submission Freeze Rules

### Code Freeze
```
⚠️ After the submission deadline:
  ☐ No new commits to the GitHub repository
  ☐ The latest commit SHA must match what's in the README
  ☐ Organizers will verify SHA before finals
```

### Model Freeze
```
⚠️ After the submission deadline:
  ☐ Model files on Google Drive must NOT be modified
  ☐ Must NOT be re-uploaded (even at the same link)
  ☐ SHA-256 checksum will be verified by organizers
```

### Consequences of Violation
- Mismatch between recorded SHA and actual code → **penalty or disqualification**
- Modified model after deadline → **penalty or disqualification**
- Code doesn't reproduce submitted images → **disqualification**

---

## 8.6 Final Round Preparation (if shortlisted)

```
☐ Presentation ready (architecture, design choices, performance)
☐ Ready for technical Q&A / viva
☐ Same code version as submitted (verified by commit SHA)
☐ Same model as submitted (verified by SHA-256)
☐ Can denoise 20 new images (481–500) in front of judges
☐ Understand every part of your solution
```
