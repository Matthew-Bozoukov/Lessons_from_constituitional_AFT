# ABOUTME: Renders an HTML slide to a 1600x900 (@2x) PNG with headless Chrome.
# ABOUTME: Run: python3 scratch/canary/render_slide.py <abs html path> <out.png>
import subprocess, sys

src, out = sys.argv[1], sys.argv[2]
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run(
    [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=2",
        "--window-size=1600,900",
        "--virtual-time-budget=5000",
        f"--screenshot={out}",
        f"file://{src}",
    ],
    check=True,
    capture_output=True,
    timeout=120,
)
print(out)
