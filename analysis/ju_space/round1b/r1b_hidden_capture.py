"""Passive last-prompt-token block outputs from existing K1/K3a generation.

    with PrefillCapture(model, n_requests) as capture:
        answers = KB.decode(..., setup=capture.setup, on_fwd=capture.on_fwd)
        hidden, layer_ids = capture.result()

No generation, persistence, random draws, precision changes or tensor mutations
are performed here. Each selected layer hook removes itself after its first call;
on_fwd(step=0) combines those small detached GPU copies and transfers once per
selected batch to CPU. Full sequences and decode-token states are never retained.
"""
import contextlib
import time

FLAG = "R1B_CAPTURE_HIDDEN"
COUNTS = {"s3": 128, "adet": 64, "ne": 64}
SCHEMA = "prefill_v1"


def _layers_of(model):
    # Same supported causal-LM wrapper paths as r1b_core.layers_of, without
    # importing the scientific pipeline or loading torch at module import.
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers", "model.model.layers"):
        value = model
        try:
            for name in path.split("."):
                value = getattr(value, name)
            layers = tuple(value)
        except (AttributeError, TypeError):
            continue
        if layers and all(hasattr(layer, "register_forward_hook") for layer in layers):
            return layers
    raise ValueError("prefill capture requires a nonempty decoder block list")


class PrefillCapture:
    """Observe the first n_requests rows, in KB.decode's original batch order.

    ``setup(b0, B)`` and ``on_fwd(b0, step)`` implement KB.decode's callbacks.
    ``result()`` returns (CPU float32 tensor [N, L, D], zero-based layer_ids).
    ``copy_seconds`` measures wall time for stacking and CPU transfer, including
    any device wait, after the forward call and before CPU validation. It is
    not a measurement of isolated capture overhead or a GPU benchmark.
    """

    def __init__(self, model, n_requests):
        import torch
        if isinstance(n_requests, bool) or not isinstance(n_requests, int) or n_requests < 0:
            raise ValueError("n_requests must be a nonnegative integer")
        self._torch = torch
        self.n_requests = n_requests
        self._layers = _layers_of(model)
        if len({id(layer) for layer in self._layers}) != len(self._layers):
            raise ValueError("prefill capture requires distinct decoder block modules")
        self.layer_ids = tuple(range(len(self._layers)))
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "get_text_config"):
            cfg = cfg.get_text_config()
        width = getattr(cfg, "hidden_size", None)
        self._width = width if isinstance(width, int) and width > 0 else None
        self._handles = {}
        self._batch = None
        self._chunks = []
        self._count = 0
        self._closed = False
        self._failed = False
        self.copy_seconds = 0.0

    def __enter__(self):
        if self._closed:
            raise RuntimeError("prefill capture is already closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._failed = True
        self.close()

    def _remove_hooks(self):
        handles, self._handles = self._handles, {}
        for handle in handles.values():
            handle.remove()

    def close(self):
        """Remove only our own hooks, including after a failed model forward."""
        if self._batch is not None:
            self._failed = True
        self._remove_hooks()
        self._batch = None
        self._closed = True

    def _hook(self, layer_id):
        def observe(module, inputs, output):
            # Always return None: the original output object is unchanged.
            try:
                batch = self._batch
                if batch is None:
                    raise RuntimeError("prefill hook called outside its batch context")
                x = output[0] if isinstance(output, (tuple, list)) and output else output
                if not isinstance(x, self._torch.Tensor) or x.ndim != 3:
                    raise ValueError(f"layer {layer_id}: expected residual tensor [B,T,D]")
                if x.dtype != self._torch.float32:
                    raise ValueError(f"layer {layer_id}: native residual dtype must be float32, got {x.dtype}")
                if x.shape[0] != batch["size"] or x.shape[1] < 1 or x.shape[2] < 1:
                    raise ValueError(f"layer {layer_id}: unexpected residual shape {tuple(x.shape)}")
                if self._width is not None and x.shape[2] != self._width:
                    raise ValueError(f"layer {layer_id}: residual width differs from model configuration")
                self._width = x.shape[2]
                # KB.decode left-pads, so -1 is the final actual prompt token
                # for every row. Copy only the selected leading request rows.
                batch["states"][layer_id] = x[:batch["selected"], -1, :].detach().clone()
            finally:
                handle = self._handles.pop(layer_id, None)
                if handle is not None:
                    handle.remove()
        return observe

    @contextlib.contextmanager
    def setup(self, b0, B):
        if self._closed or self._failed:
            raise RuntimeError("prefill capture is closed or failed")
        if (isinstance(b0, bool) or not isinstance(b0, int) or b0 < 0
                or isinstance(B, bool) or not isinstance(B, int) or B <= 0):
            raise ValueError("batch offset and size must be nonnegative/positive integers")
        if self._batch is not None:
            raise RuntimeError("prefill capture cannot observe overlapping batches")
        selected = min(B, max(0, self.n_requests - b0))
        if selected == 0:
            yield self
            return
        if b0 != self._count:
            raise ValueError(f"prefill request order mismatch: expected offset {self._count}, got {b0}")
        self._batch = dict(b0=b0, size=B, selected=selected, states=[None] * len(self._layers))
        try:
            for layer_id, layer in enumerate(self._layers):
                self._handles[layer_id] = layer.register_forward_hook(self._hook(layer_id))
            yield self
            if self._batch is not None:
                self._failed = True
                raise RuntimeError("selected prefill was not completed by on_fwd(b0, 0)")
        except BaseException:
            self._failed = True
            raise
        finally:
            self._remove_hooks()
            self._batch = None

    def on_fwd(self, b0, step):
        """Collect once after prefill; later decode callbacks perform no work."""
        if step != 0 or b0 >= self.n_requests:
            return
        batch = self._batch
        if batch is None or batch["b0"] != b0:
            raise RuntimeError("prefill callback has no matching selected batch")
        self._remove_hooks()
        try:
            if any(x is None for x in batch["states"]):
                raise ValueError("prefill did not produce every requested layer output")
            shape = (batch["selected"], self._width)
            if any(tuple(x.shape) != shape for x in batch["states"]):
                raise ValueError("prefill layer outputs have inconsistent shapes")
            t0 = time.perf_counter()
            try:
                cpu = self._torch.stack(batch["states"], dim=1).to(device="cpu")
            finally:
                self.copy_seconds += time.perf_counter() - t0
            if cpu.dtype != self._torch.float32 or not bool(self._torch.isfinite(cpu).all()):
                raise ValueError("prefill capture requires finite native float32 residuals")
            self._chunks.append(cpu)
            self._count += batch["selected"]
        except BaseException:
            self._failed = True
            raise
        finally:
            batch["states"].clear()
            self._batch = None

    def result(self):
        """Return complete observations only; never fill missing/failed requests."""
        if self._failed or self._batch is not None or self._handles:
            raise RuntimeError("prefill capture is failed or incomplete")
        if self._count != self.n_requests:
            raise ValueError(f"prefill capture incomplete: expected {self.n_requests} requests, captured {self._count}")
        torch = self._torch
        hidden = (torch.cat(self._chunks, dim=0) if self._chunks else
                  torch.empty((0, len(self._layers), self._width or 0), dtype=torch.float32, device="cpu"))
        expected = (self.n_requests, len(self._layers), self._width or 0)
        if (tuple(hidden.shape) != expected or hidden.device.type != "cpu" or hidden.dtype != torch.float32
                or hidden.requires_grad or not bool(torch.isfinite(hidden).all())):
            raise ValueError("prefill capture result failed shape/dtype/finiteness validation")
        return hidden, self.layer_ids
