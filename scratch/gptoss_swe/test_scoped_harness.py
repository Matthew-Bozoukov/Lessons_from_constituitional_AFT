# ABOUTME: Test selected-only image inventory and rejection of unexpected queries without Docker.
# ABOUTME: Run in the pinned harness env: python -m unittest scratch.gptoss_swe.test_scoped_harness.
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scratch.gptoss_swe.scoped_harness import install_scope


class ScopeTest(unittest.TestCase):
    def test_scope(self):
        try:
            from docker.models.images import ImageCollection
        except ModuleNotFoundError as error:
            if error.name != "docker":
                raise
            self.skipTest("Run this check in the locked official SWE harness environment")
        from swebench.harness import docker_utils, reporting, run_evaluation
        from swebench.harness.test_spec import test_spec

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            for obj, key in [
                (ImageCollection, "list"),
                (docker_utils, "list_images"),
                (reporting, "list_images"),
                (run_evaluation, "list_images"),
            ]:
                stack.enter_context(patch.object(obj, key, getattr(obj, key)))
            stack.enter_context(
                patch.object(
                    test_spec,
                    "make_test_spec",
                    lambda r, namespace: SimpleNamespace(
                        instance_image_key=namespace + "/" + r["instance_id"]
                    ),
                )
            )
            path = Path(temporary) / "dataset.json"
            path.write_text(
                json.dumps([{"instance_id": i} for i in ("a", "b", "unrelated")])
            )
            request = dict(
                cache_level="instance",
                clean=False,
                dataset_name=str(path),
                instance_ids=["a", "b"],
                namespace="test",
            )
            self.assertEqual(install_scope(request), ["test/a", "test/b"])
            touched = []

            def get(key):
                touched.append(key)
                return SimpleNamespace(tags=[key + ":latest"])

            images = SimpleNamespace(get=get)
            self.assertEqual(len(ImageCollection.list(images, all=True)), 2)
            self.assertEqual(
                reporting.list_images(SimpleNamespace(images=images)),
                {"test/a:latest", "test/b:latest"},
            )
            self.assertNotIn("test/unrelated", touched)
            with self.assertRaises(AssertionError):
                ImageCollection.list(images, filters={"dangling": True})
            request["instance_ids"] = ["missing"]
            with self.assertRaises(AssertionError):
                install_scope(request)
