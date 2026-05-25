
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch


@dataclass
class ReceiverProxyResult:
    ber: float = 0.0
    ser: float = 0.0
    evm: float = 0.0


# ---------------------------------------------------------------------------
# Constellation helpers (Track 4)
# ---------------------------------------------------------------------------

_SUPPORTED_MODULATIONS = ("bpsk", "qpsk", "16qam", "64qam")


def _qam_levels_gray(bits_per_dim: int) -> List[int]:
    """Return PAM levels (odd integers) in Gray-code order for one I/Q axis.

    For 4-PAM (bits_per_dim=2) the Gray-coded sequence of levels is
    [-3, -1, +3, +1], giving Gray bit mapping 00, 01, 11, 10.
    """
    num_levels = 1 << bits_per_dim
    levels_natural = [2 * idx - (num_levels - 1) for idx in range(num_levels)]
    gray_levels: List[int] = [0] * num_levels
    for idx in range(num_levels):
        gray_code = idx ^ (idx >> 1)
        gray_levels[gray_code] = levels_natural[idx]
    return gray_levels


def _build_qam_table(modulation: str):
    """Return (constellation_complex, bits_per_symbol, bit_patterns_np).

    - `constellation_complex`: complex tensor of shape (M,) with unit average power.
    - `bit_patterns`: integer tensor of shape (M, bits_per_symbol) Gray-coded.
    """
    mod = modulation.lower()
    if mod == "bpsk":
        constellation = torch.tensor([-1.0 + 0.0j, 1.0 + 0.0j], dtype=torch.complex64)
        bits = torch.tensor([[0], [1]], dtype=torch.int64)
        return constellation, 1, bits
    if mod == "qpsk":
        points = torch.tensor(
            [(-1 - 1j), (-1 + 1j), (1 - 1j), (1 + 1j)], dtype=torch.complex64,
        ) / math.sqrt(2.0)
        bits = torch.tensor(
            [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.int64,
        )
        return points, 2, bits
    if mod in ("16qam", "64qam"):
        bits_per_dim = 2 if mod == "16qam" else 3
        levels = _qam_levels_gray(bits_per_dim)
        num_levels = len(levels)
        avg_power = sum(l ** 2 for l in levels) / num_levels * 2  # I + Q
        norm = 1.0 / math.sqrt(avg_power)
        points: List[complex] = []
        bit_patterns: List[List[int]] = []
        for q_idx, q_val in enumerate(levels):
            for i_idx, i_val in enumerate(levels):
                points.append(complex(i_val * norm, q_val * norm))
                i_bits = [(i_idx >> b) & 1 for b in reversed(range(bits_per_dim))]
                q_bits = [(q_idx >> b) & 1 for b in reversed(range(bits_per_dim))]
                bit_patterns.append(i_bits + q_bits)
        constellation = torch.tensor(points, dtype=torch.complex64)
        bits = torch.tensor(bit_patterns, dtype=torch.int64)
        return constellation, 2 * bits_per_dim, bits
    raise ValueError(f"Unsupported modulation: {modulation}")


class QamReceiverProxy:
    """
    Single-tap flat-fading receiver proxy supporting {BPSK, QPSK, 16QAM, 64QAM}.

    Pipeline per pixel of the (real-valued) channel tensor `h`:
        1. Sample a complex symbol `x` from the constellation (Gray-coded).
        2. y = h * x + n, with n complex-AWGN, sigma^2 = 1/SNR_linear (symbol
           power is normalised to 1).
        3. Equalise `x_hat = y / h_est_safe` where h_est_safe keeps the sign of
           h_est and clamps its magnitude at `equalizer_eps` to avoid div-by-0.
        4. Hard-decide by nearest-constellation-point.
        5. Map decided index back to bits via the Gray table.

    LIMITATIONS (must be reported in output):
      - Channel tensor is real-valued in this dataset; complex channel h+j0 is
        assumed. This is a PROXY, not a full OFDM receiver.
      - No channel coding, interleaving, OFDM IFFT/FFT or pilot-based LS/MMSE.
      - Noise is symbol-power-normalised, not frame/RMS-normalised.
    """

    def __init__(
        self,
        equalizer_eps: float = 1e-3,
        symbol_seed: int = 1234,
    ) -> None:
        self.equalizer_eps = float(equalizer_eps)
        self.symbol_seed = int(symbol_seed)

    def _sample_symbols(
        self,
        shape: torch.Size,
        constellation: torch.Tensor,
        bit_table: torch.Tensor,
        device: torch.device,
    ):
        """Sample complex symbols and their bit labels for every pixel."""
        gen = torch.Generator(device=device)
        gen.manual_seed(self.symbol_seed)
        indices = torch.randint(
            low=0, high=constellation.numel(),
            size=shape, generator=gen, device=device,
        )
        symbols = constellation.to(device)[indices]
        bits = bit_table.to(device)[indices]
        return indices, symbols, bits

    def _awgn_complex(self, x: torch.Tensor, snr_db: float) -> torch.Tensor:
        snr_lin = 10.0 ** (snr_db / 10.0)
        sigma = math.sqrt(0.5 / snr_lin)
        noise_r = torch.randn_like(x.real) * sigma
        noise_i = torch.randn_like(x.imag) * sigma
        return x + torch.complex(noise_r, noise_i)

    def _safe_equalize_complex(
        self, y: torch.Tensor, h_est_complex: torch.Tensor,
    ) -> torch.Tensor:
        mag = torch.clamp(h_est_complex.abs(), min=self.equalizer_eps)
        phase = torch.where(
            h_est_complex.abs() > 0,
            h_est_complex / h_est_complex.abs().clamp_min(1e-12),
            torch.ones_like(h_est_complex),
        )
        denom = mag * phase
        return y / denom

    def evaluate_with_snr(
        self,
        h_true: torch.Tensor,
        h_est: torch.Tensor,
        snr_db: float,
        modulation: str,
    ) -> ReceiverProxyResult:
        """Run one (SNR, modulation) sweep point; return BER/SER/EVM."""
        device = h_true.device
        if not torch.is_complex(h_true):
            h_true = torch.complex(h_true.float(), torch.zeros_like(h_true).float())
        if not torch.is_complex(h_est):
            h_est = torch.complex(h_est.float(), torch.zeros_like(h_est).float())

        constellation, bits_per_symbol, bit_table = _build_qam_table(modulation)
        indices, symbols, bits_true = self._sample_symbols(
            h_true.shape, constellation, bit_table, device,
        )

        tx = h_true * symbols
        rx = self._awgn_complex(tx, snr_db)
        x_hat = self._safe_equalize_complex(rx, h_est)

        flat_x = x_hat.reshape(-1)
        flat_const = constellation.to(device)
        distances = (flat_x.unsqueeze(1) - flat_const.unsqueeze(0)).abs()
        decided_idx = torch.argmin(distances, dim=1)
        decided_idx = decided_idx.reshape(indices.shape)
        bits_hat = bit_table.to(device)[decided_idx]

        ser = float((decided_idx != indices).float().mean().item())
        ber = float((bits_hat != bits_true).float().mean().item())
        symbol_power = (symbols.real ** 2 + symbols.imag ** 2).mean().clamp_min(1e-12)
        err_power = ((x_hat - symbols).real ** 2 + (x_hat - symbols).imag ** 2).mean()
        evm = float(torch.sqrt(err_power / symbol_power).item())
        return ReceiverProxyResult(ber=ber, ser=ser, evm=evm)


def sweep_snr_modulation(
    h_true: torch.Tensor,
    h_est: torch.Tensor,
    snr_list: Sequence[float],
    modulation_list: Sequence[str],
    equalizer_eps: float = 1e-3,
    symbol_seed: int = 1234,
) -> List[Dict[str, float]]:
    """Run the QAM proxy across SNR × modulation; return a flat list of dicts."""
    proxy = QamReceiverProxy(equalizer_eps=equalizer_eps, symbol_seed=symbol_seed)
    rows: List[Dict[str, float]] = []
    for modulation in modulation_list:
        if modulation.lower() not in _SUPPORTED_MODULATIONS:
            raise ValueError(f"Unsupported modulation: {modulation}")
        for snr_db in snr_list:
            res = proxy.evaluate_with_snr(h_true, h_est, float(snr_db), modulation)
            rows.append(
                {
                    "modulation": modulation.lower(),
                    "snr_db": float(snr_db),
                    "ber": res.ber,
                    "ser": res.ser,
                    "evm": res.evm,
                }
            )
    return rows


class DownstreamReceiverProxy:
    """
    Receiver proxy đơn giản cho giai đoạn tìm tín hiệu.
    Giả sử tensor channel là thực-valued và dùng BPSK real-domain.
    Không thay thế equalization BER thật trong bản báo cuối.
    """

    def __init__(self, snr_db: float = 12.0, equalizer_eps: float = 1e-3, symbol_seed: int = 1234) -> None:
        self.snr_db = float(snr_db)
        self.equalizer_eps = float(equalizer_eps)
        self.symbol_seed = int(symbol_seed)

    def _make_symbols(self, shape, device: torch.device):
        gen = torch.Generator(device=device)
        gen.manual_seed(self.symbol_seed)
        bits = torch.randint(low=0, high=2, size=shape, generator=gen, device=device)
        symbols = bits.float() * 2.0 - 1.0
        return bits, symbols

    def _awgn(self, x: torch.Tensor) -> torch.Tensor:
        snr_lin = 10.0 ** (self.snr_db / 10.0)
        power = x.pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        noise_var = power / snr_lin
        noise = torch.randn_like(x) * torch.sqrt(noise_var)
        return x + noise

    def _safe_equalize(self, y: torch.Tensor, h_est: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(h_est)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        denom = sign * torch.clamp(h_est.abs(), min=self.equalizer_eps)
        return y / denom

    def evaluate(self, h_true: torch.Tensor, h_est: torch.Tensor) -> ReceiverProxyResult:
        device = h_true.device
        bits, symbols = self._make_symbols(h_true.shape, device=device)
        y = self._awgn(h_true * symbols)
        x_hat = self._safe_equalize(y, h_est)
        bits_hat = (x_hat >= 0).long()
        ber = float((bits_hat != bits).float().mean().item())
        ser = ber
        evm = torch.sqrt(((x_hat - symbols) ** 2).mean() / (symbols ** 2).mean().clamp_min(1e-8))
        return ReceiverProxyResult(ber=ber, ser=ser, evm=float(evm.item()))

    def evaluate_pair(self, h_true: torch.Tensor, h_est_clean: torch.Tensor, h_est_triggered: torch.Tensor) -> Dict[str, float]:
        clean = self.evaluate(h_true, h_est_clean)
        trig = self.evaluate(h_true, h_est_triggered)
        return {
            "proxy_ber_clean": clean.ber,
            "proxy_ber_triggered": trig.ber,
            "proxy_ber_gap": trig.ber - clean.ber,
            "proxy_ser_clean": clean.ser,
            "proxy_ser_triggered": trig.ser,
            "proxy_ser_gap": trig.ser - clean.ser,
            "proxy_evm_clean": clean.evm,
            "proxy_evm_triggered": trig.evm,
            "proxy_evm_gap": trig.evm - clean.evm,
        }
