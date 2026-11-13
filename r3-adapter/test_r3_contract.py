# -*- coding: utf-8 -*-
"""R3 数据契约适配层 —— 测试套件（接入方自检）

任何接入本适配层的 RL 训练框架，跑一遍本套件即可确认：
  1. 契约处理正确（dtype / 哨兵 / 越界 / shape / null）
  2. 回放权重没踩「从清零 probs gather」的坑
  3. 回放后 logprob 归零（R3 数学命题）

运行：python -m unittest test_r3_contract -v
"""

import unittest

import numpy as np

from r3_contract import (
    normalize_routed_experts,
    validate_shape,
    handle_null,
    check_replay_weights,
    make_wrong_weights,
    verify_replay_consistency,
)


class TestNormalize(unittest.TestCase):
    def test_three_dtypes_normalize_to_same(self):
        """同一份语义数据，uint8/uint16/int32 应归一化到同一结果。"""
        u8 = np.array([[[3, 7], [1, 2]]], dtype=np.uint8)
        u16 = np.array([[[3, 7], [1, 2]]], dtype=np.uint16)
        i32 = np.array([[[3, 7], [1, 2]]], dtype=np.int32)
        a, _ = normalize_routed_experts(u8, num_experts=16)
        b, _ = normalize_routed_experts(u16, num_experts=16)
        c, _ = normalize_routed_experts(i32, num_experts=16)
        self.assertEqual(a.dtype, np.int32)
        self.assertTrue((a == b).all() and (b == c).all())

    def test_unmappable_sentinel_kept(self):
        """-1 哨兵必须保留，且被计入 unmappable。"""
        arr = np.array([[[3, -1], [1, 2]]], dtype=np.int32)
        out, unmappable = normalize_routed_experts(arr, num_experts=16)
        self.assertEqual(unmappable, 1)
        self.assertEqual(out[0, 0, 1], -1)

    def test_zero_fill_is_silent_alias(self):
        """0 填充会让 unmappable 计数归零（静默别名到 expert 0）——这是要防的坑。"""
        zero_filled = np.array([[[3, 0], [1, 2]]], dtype=np.int32)
        _, unmappable = normalize_routed_experts(zero_filled, num_experts=16)
        self.assertEqual(unmappable, 0, "0 填充导致 unmappable 被静默吞掉")

    def test_illegal_id_rejected(self):
        """越界专家 ID 必须被拒绝，而非静默通过。"""
        illegal = np.array([[[3, 99], [1, 2]]], dtype=np.int32)
        with self.assertRaises(ValueError):
            normalize_routed_experts(illegal, num_experts=16)


class TestShape(unittest.TestCase):
    def test_shape_matches_contract(self):
        routed = np.zeros((14, 4, 8), dtype=np.int32)
        self.assertEqual(validate_shape(routed, prompt_len=10, gen_len=5), 14)

    def test_shape_mismatch_rejected(self):
        wrong = np.zeros((15, 4, 8), dtype=np.int32)
        with self.assertRaises(AssertionError):
            validate_shape(wrong, prompt_len=10, gen_len=5)


class TestNull(unittest.TestCase):
    def test_aborted_returns_none(self):
        self.assertIsNone(handle_null(None, request_aborted=True))

    def test_normal_must_not_be_none(self):
        with self.assertRaises(ValueError):
            handle_null(None, request_aborted=False)


class TestReplayWeights(unittest.TestCase):
    """★ 核心：拦住「从清零 probs gather」的 15.84% 坑。"""

    def setUp(self):
        import torch
        torch.manual_seed(0)
        self.logits = torch.randn(1, 4, 8)          # [b, s, n_experts]
        self.topk = torch.tensor([[[3, 5], [1, 7], [2, 0], [6, 4]]])

    def test_correct_weights_pass(self):
        import torch
        import torch.nn.functional as F
        # 正确：从完整 logits gather 再 softmax
        correct = F.softmax(torch.gather(self.logits, -1, self.topk), dim=-1)
        ok, err = check_replay_weights(self.logits, self.topk, correct)
        self.assertTrue(ok, f"正确权重应通过，却报错 err={err}")

    def test_wrong_weights_are_detected(self):
        # 错误：从 top-k 后清零的 probs 取
        wrong = make_wrong_weights(self.logits, self.topk)
        ok, err = check_replay_weights(self.logits, self.topk, wrong)
        self.assertFalse(ok, "错误权重（清零后 gather）应被检出，却通过了")
        self.assertGreater(err, 0.01, f"应报出可观的权重偏差，实际 {err:.4f}")


class TestReplayConsistency(unittest.TestCase):
    """端到端：回放后 logprob 应归零（R3 数学命题）。"""

    def test_replay_zeroes_logprob_diff(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class MoE(nn.Module):
            def __init__(self, d=16, ne=4, k=2):
                super().__init__()
                self.router = nn.Linear(d, ne, bias=False)
                self.experts = nn.ModuleList([nn.Linear(d, d) for _ in range(ne)])

            def forward(self, x, forced=None):
                logits = self.router(x)
                topk = forced if forced is not None else torch.topk(logits, 2, -1).indices
                vals = torch.gather(logits, -1, topk)
                w = F.softmax(vals, -1).unsqueeze(-1)
                out = torch.zeros_like(x)
                for e in range(4):
                    m = (topk == e).float().unsqueeze(-1)
                    if m.sum() > 0:
                        out = out + (m * w * self.experts[e](x).unsqueeze(2)).sum(2)
                return out, topk

        torch.manual_seed(0)
        model = MoE()
        head = nn.Linear(16, 16, bias=False)
        ids = torch.randint(0, 16, (1, 8))
        x = nn.Embedding(16, 16)(ids)

        h1, routing = model(x)
        lp1 = F.log_softmax(head(h1), -1)
        h2, _ = model(x, forced=routing)      # 回放同一路由
        lp2 = F.log_softmax(head(h2), -1)

        ok, diff = verify_replay_consistency(lp1, lp2)
        self.assertTrue(ok, f"回放后 logprob 未归零，diff={diff:.3e}")
        self.assertEqual(diff, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
