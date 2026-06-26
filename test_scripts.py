"""
Tests for download_llama70b.py and run_llama70b.py
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call
import importlib
import types


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_module_as_script(path: str, env: dict | None = None):
    """Execute a script file inside a controlled environment and capture sys.exit."""
    env = env or {}
    module_globals = {"__name__": "__main__"}
    with patch.dict(os.environ, env, clear=False):
        source = open(path).read()
        try:
            exec(compile(source, path, "exec"), module_globals)
        except SystemExit as exc:
            return exc.code, module_globals
    return 0, module_globals


DOWNLOAD_SCRIPT = os.path.join(os.path.dirname(__file__), "download_llama70b.py")
RUN_SCRIPT = os.path.join(os.path.dirname(__file__), "run_llama70b.py")


# ---------------------------------------------------------------------------
# Tests: download_llama70b.py
# ---------------------------------------------------------------------------

class TestDownloadLlama70b(unittest.TestCase):

    def test_exits_without_hf_token(self):
        """Script must exit with code 1 when HF_TOKEN is not set."""
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                with patch("huggingface_hub.snapshot_download", MagicMock()):
                    source = open(DOWNLOAD_SCRIPT).read()
                    exec(compile(source, DOWNLOAD_SCRIPT, "exec"), {"__name__": "__main__"})
        self.assertEqual(ctx.exception.code, 1)

    def test_exits_without_hf_token_via_helper(self):
        env_without_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with patch.dict(os.environ, {}, clear=True):
            code, _ = _run_module_as_script(DOWNLOAD_SCRIPT)
        self.assertEqual(code, 1)

    def test_local_dir_expands_home(self):
        """LOCAL_DIR should resolve to an absolute path under the home directory."""
        expected_suffix = "qcom_model_workspace/models/Llama-3.3-70B-Instruct"
        expanded = os.path.expanduser(f"~/qcom_model_workspace/models/Llama-3.3-70B-Instruct")
        self.assertTrue(expanded.startswith("/"), "LOCAL_DIR should be absolute after expand")
        self.assertIn(expected_suffix, expanded)

    @patch("huggingface_hub.snapshot_download")
    def test_successful_download_calls_snapshot_download(self, mock_snapshot):
        """When HF_TOKEN is present, snapshot_download should be called once."""
        mock_snapshot.return_value = None
        with patch.dict(os.environ, {"HF_TOKEN": "fake-token"}, clear=False):
            code, _ = _run_module_as_script(DOWNLOAD_SCRIPT, {"HF_TOKEN": "fake-token"})
        mock_snapshot.assert_called_once()
        self.assertEqual(code, 0)

    @patch("huggingface_hub.snapshot_download", side_effect=Exception("network error"))
    def test_download_exception_exits_with_error(self, _mock_snapshot):
        """When snapshot_download raises, the script should exit with code 1."""
        code, _ = _run_module_as_script(DOWNLOAD_SCRIPT, {"HF_TOKEN": "fake-token"})
        self.assertEqual(code, 1)

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_download_called_with_correct_repo_id(self, mock_snapshot):
        """snapshot_download must target the correct Hugging Face repo."""
        mock_snapshot.return_value = None
        _run_module_as_script(DOWNLOAD_SCRIPT, {"HF_TOKEN": "fake-token"})
        _, kwargs = mock_snapshot.call_args
        repo_id = mock_snapshot.call_args[1].get("repo_id") or mock_snapshot.call_args[0][0]
        self.assertEqual(repo_id, "meta-llama/Llama-3.3-70B-Instruct")

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_download_ignores_gguf_and_onnx(self, mock_snapshot):
        """Unwanted file formats (gguf, onnx, pki) must be in ignore_patterns."""
        mock_snapshot.return_value = None
        _run_module_as_script(DOWNLOAD_SCRIPT, {"HF_TOKEN": "fake-token"})
        kwargs = mock_snapshot.call_args[1]
        patterns = kwargs.get("ignore_patterns", [])
        self.assertIn("*.gguf", patterns)
        self.assertIn("*.onnx", patterns)
        self.assertIn("*.pki", patterns)

    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_download_passes_token(self, mock_snapshot):
        """HF_TOKEN must be forwarded to snapshot_download."""
        mock_snapshot.return_value = None
        _run_module_as_script(DOWNLOAD_SCRIPT, {"HF_TOKEN": "my-secret-token"})
        kwargs = mock_snapshot.call_args[1]
        self.assertEqual(kwargs.get("token"), "my-secret-token")


# ---------------------------------------------------------------------------
# Tests: run_llama70b.py
# ---------------------------------------------------------------------------

class TestRunLlama70b(unittest.TestCase):

    def _make_mock_model(self):
        mock_model = MagicMock()
        mock_model.compile.return_value = "/tmp/fake_qpc"
        mock_model.generate.return_value = ["mocked output"]
        return mock_model

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_model_loaded_with_correct_name(self, mock_class, mock_tokenizer):
        """QEFFAutoModelForCausalLM.from_pretrained must use the expected model ID."""
        mock_class.from_pretrained.return_value = self._make_mock_model()
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        mock_class.from_pretrained.assert_called_once_with(
            "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
        )

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_compile_called_with_two_devices(self, mock_class, mock_tokenizer):
        """compile() must request num_devices=2 to target both QIDs."""
        mock_model = self._make_mock_model()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.compile.call_args
        self.assertEqual(kwargs.get("num_devices"), 2)

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_compile_called_with_16_cores(self, mock_class, mock_tokenizer):
        """compile() must request num_cores=16 (16 NSPs per QID)."""
        mock_model = self._make_mock_model()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.compile.call_args
        self.assertEqual(kwargs.get("num_cores"), 16)

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_compile_ctx_len_and_batch_size(self, mock_class, mock_tokenizer):
        """compile() must use ctx_len=4096 and batch_size=1."""
        mock_model = self._make_mock_model()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.compile.call_args
        self.assertEqual(kwargs.get("ctx_len"), 4096)
        self.assertEqual(kwargs.get("batch_size"), 1)

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_mxint8_kv_cache_enabled(self, mock_class, mock_tokenizer):
        """compile() must enable mxint8_kv_cache for memory compression."""
        mock_model = self._make_mock_model()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.compile.call_args
        self.assertTrue(kwargs.get("mxint8_kv_cache"))

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_generate_targets_both_device_ids(self, mock_class, mock_tokenizer):
        """generate() must target device_id=[0, 1]."""
        mock_model = self._make_mock_model()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.generate.call_args
        self.assertEqual(kwargs.get("device_id"), [0, 1])

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_generate_receives_tokenizer(self, mock_class, mock_tokenizer):
        """generate() must receive the tokenizer built from the same model name."""
        mock_model = self._make_mock_model()
        fake_tokenizer = MagicMock()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = fake_tokenizer

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.generate.call_args
        self.assertIs(kwargs.get("tokenizer"), fake_tokenizer)

    @patch("transformers.AutoTokenizer")
    @patch("QEfficient.QEFFAutoModelForCausalLM")
    def test_generate_sends_non_empty_prompt(self, mock_class, mock_tokenizer):
        """generate() must pass at least one non-empty prompt string."""
        mock_model = self._make_mock_model()
        mock_class.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        _run_module_as_script(RUN_SCRIPT)

        _, kwargs = mock_model.generate.call_args
        prompts = kwargs.get("prompts", [])
        self.assertTrue(len(prompts) > 0)
        self.assertTrue(all(isinstance(p, str) and p.strip() for p in prompts))


if __name__ == "__main__":
    unittest.main()
