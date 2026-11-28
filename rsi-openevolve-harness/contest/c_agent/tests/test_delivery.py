"""Credential configuration tests: no network calls or actual credentials."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

AGENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT))
from planner import configured_api_key

TEMP = AGENT/'.test-tmp'
TEMP.mkdir(exist_ok=True)


class CredentialConfigurationTests(unittest.TestCase):
    def test_explicit_environment_key_takes_precedence_over_a_file(self):
        with patch.dict(os.environ, {'OPENEVOLVE_API_KEY': 'unit-test-not-a-real-key', 'C_API_KEYS_FILE': 'absent'}, clear=True):
            self.assertEqual(configured_api_key(), 'unit-test-not-a-real-key')

    def test_explicit_json_file_is_loaded_without_requiring_old_server_paths(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            config = Path(path)/'keys.json'
            config.write_text(json.dumps({'OPENEVOLVE_API_KEY': 'unit-test-json-placeholder'}), encoding='utf-8')
            with patch.dict(os.environ, {'C_API_KEYS_FILE': str(config)}, clear=True):
                self.assertEqual(configured_api_key(), 'unit-test-json-placeholder')

    def test_missing_example_or_whitespace_key_is_rejected_before_any_request(self):
        with patch.dict(os.environ, {'C_API_KEYS_FILE': str(TEMP/'absent.json')}, clear=True):
            with self.assertRaises(ValueError): configured_api_key()
        for value in ['REPLACE_WITH_YOUR_KEY', 'contains whitespace']:
            with patch.dict(os.environ, {'OPENEVOLVE_API_KEY': value}, clear=True):
                with self.assertRaises(ValueError): configured_api_key()


if __name__ == '__main__':
    unittest.main()
