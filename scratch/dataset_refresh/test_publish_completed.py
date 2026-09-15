# ABOUTME: Offline regression checks for completed-corpus publication boundaries.
# ABOUTME: Exercises audit scoping, flattened evidence, quota validation and tamper refusal without network calls.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scratch.dataset_refresh import publish_completed as pub


class PublicationBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)

    def ledger(self, entries):
        pub.write_json(self.root / 'budget/spend.json', entries)
        return {'budget_root': str(self.root / 'budget')}

    def entry(self, **changes):
        return {'run_root': str(self.root), 'arm': 'nonmoral-advice', 'status': 'completed', **changes}

    def test_shared_accounting_does_not_import_other_run_raw_calls(self):
        entries = [self.entry(), self.entry(run_root='old-run'), self.entry(arm='other-arm')]
        _, shared, scoped = pub.scoped_ledger(self.ledger(entries), self.root, 'nonmoral-advice', 3)
        self.assertEqual(len(shared), 3)
        self.assertEqual(scoped, entries[:1])

    def test_cutoff_cannot_hide_later_arm_calls(self):
        with self.assertRaisesRegex(ValueError, 'excludes calls'):
            pub.scoped_ledger(self.ledger([self.entry(), self.entry()]), self.root, 'nonmoral-advice', 1)

    def test_active_arm_calls_block_snapshot(self):
        with self.assertRaisesRegex(ValueError, 'active API'):
            pub.scoped_ledger(self.ledger([self.entry(status='reserved')]), self.root, 'nonmoral-advice', 1)

    def test_transport_header_is_not_published(self):
        with self.assertRaisesRegex(ValueError, 'transport'):
            pub.safe_audit({'request': {'headers': {'Authorization': 'fake fixture'}}})
        pub.safe_audit({'request': {'messages': [{'content': 'Explain authorization as an ordinary word.'}]}})

    def test_flatten_preserves_receipts_values_and_file_hashes(self):
        source = self.root / 'source'
        pub.write_json(source / 'records/t1_000/result.json', {'status': 'accepted'})
        pub.write_json(source / 'records/t1_000/result.receipt.json', {'sha256': 'fixture'})
        pub.write_json(source / 'config.json', {'pipeline': 'fixture'})
        pub.freeze_arm(source, self.root / 'out')
        rows = pub.read_rows(self.root / 'out/record_checkpoints.jsonl')
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['value'], pub.read_json(source / row['path']))
            self.assertEqual(row['sha256'], pub.digest((source / row['path']).read_bytes()))
        self.assertFalse((self.root / 'out/records').exists())

    def test_716_total_cannot_hide_wrong_trait_balance(self):
        rows = [{'metadata': {'trait_id': 't1'}} for _ in range(716)]
        with self.assertRaisesRegex(ValueError, 'quotas'):
            pub.validate_export(rows, {})

    def test_tampered_snapshot_refused_before_network(self):
        (self.root / 'dataset.jsonl').write_text('changed', encoding='utf-8')
        pub.write_json(self.root / 'publication_manifest.json', {'files': {'dataset.jsonl': 'old'}})
        with patch.object(pub, 'push_run_dir') as network:
            with self.assertRaisesRegex(ValueError, 'changed'):
                pub.push(self.root)
            network.assert_not_called()


if __name__ == '__main__':
    unittest.main()
