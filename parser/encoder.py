"""Pinned pretrained MiniLM sentence embeddings, using CPU ONNX inference."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

REPO = "Xenova/all-MiniLM-L6-v2"
REVISION = "cb3d680149bf9a3209564e1b27ab3bb355b65707"
FILES = {
    "onnx": ("onnx/model_quantized.onnx", "model.onnx", "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"),
    "tokenizer": ("tokenizer.json", "tokenizer.json", "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
}
_ENGINES = {}


def load_encoder_assets() -> dict:
    import requests

    root = Path(os.getenv("CARDQUERY_ENCODER_PATH", str(Path.home() / ".cache/cardquery/minilm")))
    root.mkdir(parents=True, exist_ok=True)
    result = {"repository": REPO, "revision": REVISION}
    for key, (remote, filename, checksum) in FILES.items():
        path = root / filename
        if not path.exists():
            response = requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{remote}", timeout=120)
            response.raise_for_status()
            content = response.content
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(f"Pretrained encoder checksum mismatch: {filename}")
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"Pretrained encoder checksum mismatch: {filename}")
        result[key] = content
    return result


class MiniLMEncoder:
    def __init__(self, assets):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(assets["onnx"], options, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_str(assets["tokenizer"].decode())
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.cache = {}

    def encode(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        for offset in range(0, len(missing), 32):
            batch = missing[offset:offset + 32]
            tokens = self.tokenizer.encode_batch(batch)
            ids = np.asarray([token.ids for token in tokens], dtype=np.int64)
            mask = np.asarray([token.attention_mask for token in tokens], dtype=np.int64)
            segments = np.asarray([token.type_ids for token in tokens], dtype=np.int64)
            hidden = self.session.run(None, {"input_ids": ids, "attention_mask": mask,
                                            "token_type_ids": segments})[0]
            vectors = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            self.cache.update(zip(batch, vectors.astype(np.float32)))
        return np.asarray([self.cache[text] for text in texts], dtype=np.float32)


def get_encoder(assets):
    key = assets["revision"]
    if key not in _ENGINES:
        _ENGINES[key] = MiniLMEncoder(assets)
    return _ENGINES[key]
