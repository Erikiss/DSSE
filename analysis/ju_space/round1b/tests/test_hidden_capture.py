"""Real tiny CPU-torch checks; no transformers, downloaded models or weights."""
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1b_hidden_capture import COUNTS, FLAG, SCHEMA, PrefillCapture

try:
    import torch
    from torch import nn
except ImportError:
    torch = None


if torch is not None:
    class TinyBlock(nn.Module):
        def __init__(self, width, as_tuple=False):
            super().__init__()
            self.linear = nn.Linear(width, width)
            self.as_tuple = as_tuple

        def forward(self, x):
            y = x + torch.tanh(self.linear(x))
            return (y, None) if self.as_tuple else y


    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=5)
            self.embedding = nn.Embedding(19, 5)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyBlock(5), TinyBlock(5, as_tuple=True), TinyBlock(5)])
            self.head = nn.Linear(5, 19)
            self.calls = 0
            self.fail_at = None

        def forward(self, ids, reference=False):
            self.calls += 1
            x = self.embedding(ids)
            states = []
            for i, layer in enumerate(self.model.layers):
                if self.fail_at == i:
                    raise RuntimeError("intentional forward failure")
                out = layer(x)
                x = out[0] if isinstance(out, tuple) else out
                if reference:
                    states.append(x[:, -1, :].detach().clone())
            logits = self.head(x)
            return (logits, torch.stack(states, 1)) if reference else logits


@unittest.skipIf(torch is None, "CPU torch is required; use /workspace/.venvs/htq-tests/bin/python")
class PrefillCaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TinyModel().eval()
        self.ids = torch.tensor([[0, 0, 2, 3], [0, 4, 5, 6], [7, 8, 9, 10]])

    def hook_count(self):
        return sum(len(layer._forward_hooks) for layer in self.model.model.layers)

    def test_constants_match_fixed_scope(self):
        self.assertEqual(FLAG, "R1B_CAPTURE_HIDDEN")
        self.assertEqual(SCHEMA, "prefill_v1")
        self.assertEqual(COUNTS, {"s3": 128, "adet": 64, "ne": 64})

    def test_all_layers_match_reference_and_left_padding_final_token(self):
        with torch.no_grad():
            expected_logits, expected = self.model(self.ids, reference=True)
        with PrefillCapture(self.model, 3) as capture:
            with capture.setup(0, 3), torch.no_grad():
                actual_logits = self.model(self.ids)
                # Every hook removes itself before any decode forward begins.
                self.assertEqual(self.hook_count(), 0)
                capture.on_fwd(0, 0)
            hidden, layers = capture.result()
        after_close, after_layers = capture.result()
        again, _ = capture.result()
        self.assertTrue(torch.equal(after_close, hidden))
        self.assertTrue(torch.equal(again, hidden))
        self.assertEqual(after_layers, layers)
        self.assertTrue(torch.equal(expected_logits, actual_logits))
        self.assertTrue(torch.equal(expected, hidden))
        self.assertEqual(tuple(hidden.shape), (3, 3, 5))
        self.assertEqual(layers, (0, 1, 2))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertEqual(hidden.device.type, "cpu")
        self.assertFalse(hidden.requires_grad)
        self.assertGreaterEqual(capture.copy_seconds, 0.0)
        with torch.no_grad():
            _, just_last = self.model(self.ids[:, -1:], reference=True)
        # The independent one-position forward can use a differently shaped
        # GEMM; exact identity above uses the actual padded forward shape.
        torch.testing.assert_close(hidden, just_last, rtol=1e-6, atol=1e-7)

    def test_cap_across_batches_and_no_decode_or_unselected_hooks(self):
        ids = torch.cat((self.ids, self.ids + 1, self.ids + 2), 0)
        with torch.no_grad():
            _, expected = self.model(ids, reference=True)
        self.model.calls = 0
        with PrefillCapture(self.model, 5) as capture:
            for b0 in range(0, len(ids), 4):
                rows = ids[b0:b0 + 4]
                with capture.setup(b0, len(rows)), torch.no_grad():
                    self.assertEqual(self.hook_count(), 3 if b0 < 5 else 0)
                    self.model(rows)
                    capture.on_fwd(b0, 0)
                    self.assertEqual(self.hook_count(), 0)
                    self.model(rows[:, -1:])
                    capture.on_fwd(b0, 1)
                    self.model(rows[:, -1:])
                    capture.on_fwd(b0, 2)
            hidden, _ = capture.result()
        self.assertTrue(torch.equal(hidden, expected[:5]))
        self.assertEqual(self.model.calls, 9)

    def test_logits_generation_rng_parameters_and_existing_hooks_unchanged(self):
        seen = []
        existing = self.model.model.layers[0].register_forward_hook(lambda *args: seen.append(1))
        self.addCleanup(existing.remove)
        params = {name: value.detach().clone() for name, value in self.model.state_dict().items()}

        def generate(capture=None):
            logits, answers = [], []
            for b0 in (0, 2):
                rows = self.ids[b0:b0 + 2]
                with (capture.setup(b0, len(rows)) if capture else contextlib.nullcontext()), torch.no_grad():
                    cur = self.model(rows)
                    if capture:
                        capture.on_fwd(b0, 0)
                    for step in range(3):
                        logits.append(cur.detach().clone())
                        token = torch.multinomial(cur[:, -1, :].softmax(-1), 1)
                        answers.append(token.clone())
                        if step < 2:
                            cur = self.model(token)
                            if capture:
                                capture.on_fwd(b0, step + 1)
            return logits, answers

        start = torch.random.get_rng_state().clone()
        reference_logits, reference_answers = generate()
        reference_rng = torch.random.get_rng_state().clone()
        reference_calls = self.model.calls
        self.model.calls = 0
        torch.random.set_rng_state(start)
        with PrefillCapture(self.model, 2) as capture:
            actual_logits, actual_answers = generate(capture)
            hidden, _ = capture.result()
        self.assertEqual(tuple(hidden.shape), (2, 3, 5))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference_logits, actual_logits)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(reference_answers, actual_answers)))
        self.assertTrue(torch.equal(reference_rng, torch.random.get_rng_state()))
        self.assertEqual(reference_calls, self.model.calls)
        self.assertEqual(self.hook_count(), 1)
        self.assertEqual(len(seen), 2 * reference_calls)
        self.assertTrue(all(torch.equal(params[name], value) for name, value in self.model.state_dict().items()))

    def test_failed_forward_removes_remaining_hooks_and_never_fills_results(self):
        capture = PrefillCapture(self.model, 3)
        self.model.fail_at = 1
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with capture.setup(0, 3), torch.no_grad():
                self.model(self.ids)
        self.assertEqual(self.hook_count(), 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            capture.result()
        capture.close()

    def test_native_non_fp32_is_rejected_without_casting_or_hook_leaks(self):
        self.model.double()
        capture = PrefillCapture(self.model, 1)
        with self.assertRaisesRegex(ValueError, "native residual dtype"):
            with capture.setup(0, 3), torch.no_grad():
                self.model(self.ids)
        self.assertEqual(self.hook_count(), 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            capture.result()

    def test_nonfinite_residuals_are_rejected(self):
        with torch.no_grad():
            self.model.embedding.weight[3, 0] = float("nan")
        capture = PrefillCapture(self.model, 1)
        with self.assertRaisesRegex(ValueError, "finite native float32"):
            with capture.setup(0, 3), torch.no_grad():
                self.model(self.ids)
                capture.on_fwd(0, 0)
        self.assertEqual(self.hook_count(), 0)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            capture.result()

    def test_missing_prefill_callback_and_incomplete_count_fail_clearly(self):
        capture = PrefillCapture(self.model, 3)
        with self.assertRaisesRegex(RuntimeError, "on_fwd"):
            with capture.setup(0, 3), torch.no_grad():
                self.model(self.ids)
        self.assertEqual(self.hook_count(), 0)
        with PrefillCapture(self.model, 4) as capture:
            with capture.setup(0, 3), torch.no_grad():
                self.model(self.ids)
                capture.on_fwd(0, 0)
            with self.assertRaisesRegex(ValueError, "expected 4 requests, captured 3"):
                capture.result()

    def test_zero_requests_use_no_hooks_and_return_valid_empty_tensor(self):
        with PrefillCapture(self.model, 0) as capture:
            with capture.setup(0, 3), torch.no_grad():
                self.assertEqual(self.hook_count(), 0)
                self.model(self.ids)
                capture.on_fwd(0, 0)
            hidden, layers = capture.result()
        self.assertEqual(tuple(hidden.shape), (0, 3, 5))
        self.assertEqual(layers, (0, 1, 2))

    def test_close_removes_hooks_on_outer_exception_and_is_idempotent(self):
        capture = PrefillCapture(self.model, 3)
        with self.assertRaisesRegex(RuntimeError, "outside"):
            with capture:
                with capture.setup(0, 3):
                    self.assertEqual(self.hook_count(), 3)
                    raise RuntimeError("outside")
        self.assertEqual(self.hook_count(), 0)
        capture.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with capture.setup(0, 3):
                pass

    def test_request_order_and_invalid_counts_are_rejected(self):
        for n in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                PrefillCapture(self.model, n)
        with PrefillCapture(self.model, 3) as capture:
            with self.assertRaisesRegex(ValueError, "order mismatch"):
                with capture.setup(1, 2):
                    pass
        self.assertEqual(self.hook_count(), 0)


if __name__ == "__main__":
    unittest.main()
