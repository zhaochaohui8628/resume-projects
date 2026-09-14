"""pytorch-crf 接入校验：**不自实现 CRF**，只验证「库调用 + 标签/掩码规整」正确。

判据
----
1. 库返回的对数似然 == 暴力枚举算出的 `gold_path_score − logZ`；
2. 库的 Viterbi 解码 == 暴力枚举的最优标签序列；
3. 掩码生效（pad 位置不计分、不计入配分函数）；
4. `path_logprob_per_token` 与「对数似然 / 有效长度」一致。
"""
from __future__ import annotations

import itertools

import torch

from ner2.src.bert.crf import CRFHead, import_crf_class


def _random_crf(num_tags: int, seed: int = 0):
    torch.manual_seed(seed)
    head = CRFHead(num_tags)
    inner = head.crf
    with torch.no_grad():
        inner.transitions.copy_(torch.randn(num_tags, num_tags) * 0.7)
        inner.start_transitions.copy_(torch.randn(num_tags) * 0.7)
        inner.end_transitions.copy_(torch.randn(num_tags) * 0.7)
    return head, inner


def _brute_force(emissions: torch.Tensor, inner, seq_len: int | None = None):
    """枚举全部标签序列的得分 → (seqs, scores, logZ)。"""
    C = emissions.shape[2]
    T = emissions.shape[1] if seq_len is None else seq_len
    start = inner.start_transitions.detach()
    trans = inner.transitions.detach()
    end = inner.end_transitions.detach()
    em = emissions[0].detach()
    seqs, scores = [], []
    for s in itertools.product(range(C), repeat=T):
        sc = float(start[s[0]]) + float(em[0, s[0]])
        for t in range(1, T):
            sc += float(em[t, s[t]]) + float(trans[s[t - 1], s[t]])
        sc += float(end[s[-1]])
        seqs.append(s)
        scores.append(sc)
    logZ = float(torch.logsumexp(torch.tensor(scores), dim=0))
    return seqs, scores, logZ


def test_library_available():
    CRF = import_crf_class()
    assert CRF is not None
    head = CRFHead(5)
    assert hasattr(head.crf, "transitions")


def test_loglik_matches_bruteforce():
    C, T = 3, 5
    head, inner = _random_crf(C, seed=1)
    emissions = torch.randn(1, T, C)
    mask = torch.ones(1, T, dtype=torch.bool)
    tags = torch.tensor([[0, 2, 1, 0, 2]])
    llh = float(head.loglik(emissions, tags, mask, reduction="none")[0])
    seqs, scores, logZ = _brute_force(emissions, inner)
    gold = scores[seqs.index(tuple(tags[0].tolist()))]
    assert abs(llh - (gold - logZ)) < 1e-4
    # nll = -llh
    nll = float(head.nll(emissions, tags, mask, reduction="none")[0])
    assert abs(nll + llh) < 1e-6


def test_viterbi_matches_bruteforce():
    C, T = 3, 5
    head, inner = _random_crf(C, seed=2)
    emissions = torch.randn(1, T, C)
    mask = torch.ones(1, T, dtype=torch.bool)
    seqs, scores, _ = _brute_force(emissions, inner)
    best = list(seqs[int(torch.tensor(scores).argmax())])
    assert list(head.decode(emissions, mask)[0]) == best


def test_mask_excludes_padding():
    C, T = 3, 5
    head, inner = _random_crf(C, seed=3)
    emissions = torch.randn(1, T, C)
    mask = torch.ones(1, T, dtype=torch.bool)
    mask[0, -2:] = False
    tags = torch.tensor([[0, 2, 1, 1, 1]])
    llh = float(head.loglik(emissions, tags, mask, reduction="none")[0])
    seqs, scores, logZ = _brute_force(emissions, inner, seq_len=3)
    gold = scores[seqs.index((0, 2, 1))]
    assert abs(llh - (gold - logZ)) < 1e-4
    # 解码长度 == 有效长度
    assert len(head.decode(emissions, mask)[0]) == 3


def test_path_logprob_per_token():
    C, T = 3, 6
    head, _ = _random_crf(C, seed=4)
    emissions = torch.randn(2, T, C)
    mask = torch.ones(2, T, dtype=torch.bool)
    mask[1, 4:] = False
    tags = torch.zeros(2, T, dtype=torch.long)
    per_tok = head.path_logprob_per_token(emissions, tags, mask)
    llh = head.loglik(emissions, tags, mask, reduction="none")
    lengths = mask.sum(-1)
    assert per_tok.shape == (2,)
    for b in range(2):
        assert abs(float(per_tok[b]) - float(llh[b] / lengths[b])) < 1e-5
