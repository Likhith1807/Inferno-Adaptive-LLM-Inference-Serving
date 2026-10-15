"""
The AWQ -> kernel-layout repack is the riskiest step of the vLLM integration: a wrong nibble order gives plausible
but wrong logits. Verified here on the CPU against an independent, deliberately naive dequantization.
"""
import pytest

torch = pytest.importorskip("torch")

from inferno.kernels import vllm_int4  # noqa: E402

AWQ_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _pack_awq(vals: "torch.Tensor") -> "torch.Tensor":
    """[R, N] values 0..15 -> [R, N/8] int32, nibble i of word c holds channel 8c + AWQ_ORDER[i] (AutoAWQ layout)."""
    R, N = vals.shape
    out = torch.zeros((R, N // 8), dtype=torch.int64)
    for c in range(N // 8):
        for i in range(8):
            out[:, c] |= vals[:, 8 * c + AWQ_ORDER[i]].to(torch.int64) << (4 * i)
    return torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)


def test_unpack_inverts_the_awq_packing():
    g = torch.Generator().manual_seed(0)
    vals = torch.randint(0, 16, (5, 24), generator=g)
    assert torch.equal(vllm_int4._unpack_awq(_pack_awq(vals)).to(torch.int64), vals.to(torch.int64))


def test_repacked_layout_dequantizes_to_the_same_weights():
    g = torch.Generator().manual_seed(1)
    K, N, G = 64, 16, 32
    q = torch.randint(0, 16, (K, N), generator=g)
    z = torch.randint(3, 12, (K // G, N), generator=g)
    s = (torch.rand((K // G, N), generator=g) * 0.02 + 0.004).to(torch.float16)
    W, S, Z = vllm_int4.repack_awq(_pack_awq(q), _pack_awq(z), s)
    assert W.shape == (N, K // 8) and W.dtype == torch.int32
    assert S.shape == (N, K // G) and Z.shape == (N, K // G)

    # dequantize from the kernel layout: nibble j of word w holds k = 8w + j of output channel n
    words = W.to(torch.int64) & 0xFFFFFFFF
    got = torch.zeros((N, K), dtype=torch.float32)
    for n in range(N):
        for k in range(K):
            qq = (words[n, k // 8] >> (4 * (k % 8))) & 0xF
            got[n, k] = (qq.item() - Z[n, k // G].item()) * S[n, k // G].item()
    want = ((q.to(torch.float32) - z.repeat_interleave(G, dim=0).to(torch.float32)) * s.repeat_interleave(G, dim=0).to(torch.float32)).t()
    assert torch.allclose(got, want, atol=1e-3)


def test_high_bit_words_survive_the_int32_wraparound():
    q = torch.full((32, 8), 15, dtype=torch.int64)          # every nibble 0xF -> word 0xFFFFFFFF -> int32 -1
    z = torch.zeros((1, 8), dtype=torch.int64)
    s = torch.ones((1, 8), dtype=torch.float16)
    W, _, _ = vllm_int4.repack_awq(_pack_awq(q), _pack_awq(z), s)
    assert (W == -1).all()
