# ABOUTME: Verify the billing watchdog cannot stop unrelated or unexpired CPU instances.
# ABOUTME: Uses a fake provider; no credentials, rentals, or network calls.
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scratch import swebench_cpu_watchdog as watchdog


class WatchdogTest(unittest.TestCase):
    def invoke(self, *, expired=True, label="nika-test", status="running", check=False, boot_expired=False):
        client = Mock()
        client.show_instance.return_value = {"id": 42, "label": label, "actual_status": status}
        client.stop_instance.return_value = {"success": True}
        now = datetime.now(timezone.utc)
        receipt = {"instance_id": 42, "label": "nika-test",
                   "stop_at": (now + timedelta(hours=-1 if expired else 1)).isoformat()}
        if boot_expired:
            receipt["boot_deadline"] = (now - timedelta(minutes=1)).isoformat()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "receipt.json"
            p.write_text(json.dumps(receipt))
            argv = ["watchdog", "--receipt", str(p), "--env", "unused"] + (["--check"] if check else [])
            with patch.object(watchdog, "VastAI", return_value=client), \
                 patch.object(watchdog, "dotenv_values", return_value={"VAST_API_KEY": "fake"}), \
                 patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                watchdog.main()
        return client

    def test_expired_stops_exact_id(self):
        self.invoke().stop_instance.assert_called_once_with(42)

    def test_not_expired_does_nothing(self):
        self.invoke(expired=False).stop_instance.assert_not_called()

    def test_check_is_read_only_even_after_expiry(self):
        self.invoke(check=True).stop_instance.assert_not_called()

    def test_already_stopped_is_idempotent(self):
        self.invoke(status="stopped").stop_instance.assert_not_called()

    def test_changed_ownership_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Ownership"):
            self.invoke(label="someone-else")

    def test_failed_boot_stops_early(self):
        self.invoke(expired=False, boot_expired=True).stop_instance.assert_called_once_with(42)
