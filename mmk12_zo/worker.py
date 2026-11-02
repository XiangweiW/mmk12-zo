"""In-place bf16 perturbations for vanilla ARS and ordinary ES.

Noise is rebuilt from a per-tensor seed. The same approximation as ES at Scale
is used: add noise, evaluate, subtract it. The parameter update accumulates
in fp32 one tensor at a time and is written back to bf16.
"""

from __future__ import annotations

import math
import os
import time

import torch

from mmk12_zo.seeds import tensor_seed


class ZOExtension:
    def load_weights_from(self, path: str) -> dict:
        self._active = None
        if os.path.isdir(path):
            source = "pretrained-engine"
            loaded = 0
        else:
            from safetensors.torch import load_file

            loaded = self._copy_saved(load_file(path, device="cpu"))
            source = path
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "loaded": loaded,
            "source": source,
            "n_params": sum(1 for _ in self._iter_params()),
            "checksum": self._checksum(),
            "caches_cleared": self.clear_caches(),
        }

    def set_weights(self, direction_seed: int, sign: int, noise_std: float) -> dict:
        started = time.perf_counter()
        self._restore_active()
        applied = False
        if sign != 0 and noise_std != 0.0:
            self._apply_noise(int(direction_seed), int(sign), float(noise_std))
            self._active = (int(direction_seed), int(sign), float(noise_std))
            applied = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {
            "applied": applied,
            "perturb_seconds": time.perf_counter() - started,
            "caches_cleared": self.clear_caches(),
        }

    def apply_update(self, directions: list[list[float]], scale: float) -> dict:
        started = time.perf_counter()
        self._restore_active()
        pairs = [(int(seed), float(coeff)) for seed, coeff in directions]
        if scale != 0.0 and pairs:
            stats = self._inplace_update(pairs, float(scale))
        else:
            stats = {"l2": 0.0, "numel": 0}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats["update_seconds"] = time.perf_counter() - started
        stats["checksum"] = self._checksum()
        stats["caches_cleared"] = self.clear_caches()
        return stats

    def save_weights(self, path: str) -> dict:
        from safetensors.torch import save_file

        self._restore_active()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        save_file({name: param.detach().to("cpu") for name, param in self._iter_params()}, temporary)
        os.replace(temporary, path)
        return {"path": path, "checksum": self._checksum()}

    def clear_caches(self) -> list[str]:
        cleared = []
        for name in ("reset_encoder_cache", "reset_mm_cache"):
            method = getattr(self, name, None)
            if method is None:
                continue
            method()
            cleared.append(name)
        return cleared

    def _inplace_update(self, directions: list[tuple[int, float]], scale: float) -> dict:
        total_sq = 0.0
        total = 0
        with torch.no_grad():
            for name, param in self._iter_params():
                acc = torch.zeros(param.shape, device=param.device, dtype=torch.float32)
                for direction, coeff in directions:
                    if coeff == 0.0:
                        continue
                    noise = self._randn(param, direction, name)
                    acc.add_(noise.float(), alpha=coeff)
                    del noise
                acc.mul_(scale)
                total_sq += float(acc.pow(2).sum().item())
                total += param.numel()
                param.data.copy_(torch.add(param.data.float(), acc).to(dtype=param.dtype))
                del acc
        return {"l2": math.sqrt(total_sq), "numel": total}

    def _apply_noise(self, direction_seed: int, sign: int, noise_std: float) -> None:
        with torch.no_grad():
            for name, param in self._iter_params():
                noise = self._randn(param, direction_seed, name)
                param.data.add_(noise, alpha=float(sign) * float(noise_std))
                del noise

    def _restore_active(self) -> None:
        active = getattr(self, "_active", None)
        if not active:
            return
        direction_seed, sign, noise_std = active
        self._active = None
        self._apply_noise(direction_seed, -sign, noise_std)

    def _randn(self, param: torch.nn.Parameter, direction: int, name: str) -> torch.Tensor:
        generator = torch.Generator(device=param.device)
        generator.manual_seed(tensor_seed(direction, name))
        return torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=generator)

    def _iter_params(self):
        seen: set[int] = set()
        for name, param in self._model().named_parameters():
            if not param.is_floating_point():
                continue
            ptr = param.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            yield name, param

    def _checksum(self) -> int:
        total = 0
        for _, param in self._iter_params():
            total += int(param.detach().contiguous().view(torch.int16).sum().item())
        return total

    def _copy_saved(self, raw: dict[str, torch.Tensor]) -> int:
        copied = 0
        missing = []
        for name, param in self._iter_params():
            if name not in raw:
                missing.append(name)
                continue
            param.data.copy_(raw[name].to(device=param.device, dtype=param.dtype))
            copied += 1
        if missing:
            raise RuntimeError(f"checkpoint is missing parameters: {missing[:8]}")
        return copied

    def _model(self):
        getter = getattr(self, "get_model", None)
        if getter is not None:
            return getter()
        raise RuntimeError("vLLM worker does not expose the model")
