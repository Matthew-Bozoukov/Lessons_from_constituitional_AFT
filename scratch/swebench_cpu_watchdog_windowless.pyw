# ABOUTME: Run the Windows expiry watchdog without opening a console window.
# ABOUTME: Task Scheduler invokes this with pythonw.exe and the normal watchdog arguments.
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    receipt = Path(sys.argv[sys.argv.index('--receipt') + 1])
    log_path = receipt.parent / 'expiry-watchdog.log'
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\n' + datetime.now(timezone.utc).isoformat() + '\n')
        log.flush()
        result = subprocess.run(
            [str(Path(sys.executable).with_name('python.exe')),
             str(Path(__file__).with_name('swebench_cpu_watchdog.py')), *sys.argv[1:]],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        log.write(f'Exit code: {result.returncode}\n')
    sys.exit(result.returncode)
