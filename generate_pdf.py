"""Generate PDF from HTML report using headless Chrome/Edge."""
import subprocess
import sys
import os
from pathlib import Path

def find_browser():
    """Find Chrome or Edge executable."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def main():
    project_dir = Path(__file__).parent
    html_file = project_dir / "Noice_Floor_Ninjas_Report.html"
    pdf_file = project_dir / "Noice_Floor_Ninjas.pdf"
    
    if not html_file.exists():
        print(f"ERROR: HTML file not found: {html_file}")
        sys.exit(1)
    
    browser = find_browser()
    if not browser:
        print("ERROR: No Chrome or Edge browser found.")
        sys.exit(1)
    
    print(f"Using browser: {browser}")
    print(f"Input:  {html_file}")
    print(f"Output: {pdf_file}")
    
    # Convert to file:// URL
    html_url = html_file.as_uri()
    
    cmd = [
        browser,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-software-rasterizer",
        f"--print-to-pdf={pdf_file}",
        "--print-to-pdf-no-header",
        "--no-margins",  # We handle margins in CSS @page
        html_url,
    ]
    
    print(f"Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"Browser stderr: {result.stderr[:500]}")
    
    if pdf_file.exists():
        size_kb = pdf_file.stat().st_size / 1024
        print(f"SUCCESS: PDF generated at {pdf_file} ({size_kb:.1f} KB)")
    else:
        print("FAILED: PDF was not generated.")
        print(f"stdout: {result.stdout[:300]}")
        print(f"stderr: {result.stderr[:500]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
