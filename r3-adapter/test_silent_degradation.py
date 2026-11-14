# -*- coding: utf-8 -*-
"""silent_degradation.py 的自测

这个工具自己静默失败过两次（正则漏检），所以它必须先证明自己不会静默失败。
每条规则至少一个正例一个反例；pip 解析覆盖今天事故的真实输出格式。

运行：python -m unittest test_silent_degradation -v
"""

import unittest

import silent_degradation as sd


def codes(findings):
    return {f.code for f in findings}


def sev(findings, code):
    for f in findings:
        if f.code == code:
            return f.severity
    return None


HEALTHY = """
INFO Using 'HUMMING' WNA16 MoE backend
INFO offloaded 3.01 GB of weights to CPU
INFO Model loading took 4.83 GiB
INFO Dynamo detected batch_invariant.py:293 matmul_persistent
INFO Available KV cache 10.62 GiB
INFO Uvicorn running on http://0.0.0.0:8000
"""

BAD = """
WARNING Int4 weights are dequantized to BF16 at load time
INFO Using 'EMULATION' MoE backend
INFO offloaded 3.01 GB of weights to CPU
INFO Model loading took 27.90 GiB
WARNING Default vLLM sampling parameters have been overridden by the model's `generation_config.json`
WARNING Pinned memory is not available on this platform
WARNING Flash attention kernel not used because: dtype mismatch
"""


class TestParseFacts(unittest.TestCase):
    def test_healthy_facts(self):
        f = sd.parse_facts(HEALTHY)
        self.assertEqual(f["moe_backend"], "HUMMING")
        self.assertTrue(f["batch_invariant"])
        self.assertEqual(f["offload_gib"], 3.01)
        self.assertEqual(f["model_loading_gib"], 4.83)
        self.assertEqual(f["kv_cache_gib"], 10.62)
        self.assertFalse(f["emulation"])

    def test_bad_facts(self):
        f = sd.parse_facts(BAD)
        self.assertTrue(f["emulation"])
        self.assertTrue(f["dequant_bf16"])
        self.assertTrue(f["sampling_override"])
        self.assertTrue(f["pinned_mem_fail"])
        self.assertFalse(f["batch_invariant"])
        self.assertIn("Flash attention", f["sdpa_fallbacks"])


class TestRules(unittest.TestCase):
    def test_healthy_no_critical(self):
        f = sd.parse_facts(HEALTHY)
        r = sd.run_rules(f, HEALTHY, model_gib=7.84)
        self.assertNotIn("CRITICAL", {x.severity for x in r})
        self.assertIn("OFFLOAD_OK", codes(r))
        self.assertIn("BATCH_INVARIANT_OK", codes(r))

    def test_emulation_critical(self):
        f = sd.parse_facts(BAD)
        r = sd.run_rules(f, BAD)
        self.assertEqual(sev(r, "QUANT_EMULATION"), "CRITICAL")

    def test_offload_noop_with_baseline(self):
        f = sd.parse_facts(BAD)
        r = sd.run_rules(f, BAD, model_gib=7.84)
        self.assertEqual(sev(r, "OFFLOAD_NOOP"), "CRITICAL")

    def test_offload_unverified_without_baseline(self):
        f = sd.parse_facts(BAD)
        r = sd.run_rules(f, BAD)  # 不给 model_gib
        self.assertEqual(sev(r, "OFFLOAD_UNVERIFIED"), "WARN")
        self.assertNotIn("OFFLOAD_NOOP", codes(r))

    def test_batch_invariant_missing(self):
        f = sd.parse_facts(BAD)
        r = sd.run_rules(f, BAD)
        self.assertEqual(sev(r, "BATCH_INVARIANT_MISSING"), "CRITICAL")

    def test_sampling_and_pinned_warn(self):
        f = sd.parse_facts(BAD)
        r = sd.run_rules(f, BAD)
        self.assertEqual(sev(r, "SAMPLING_OVERRIDE"), "WARN")
        self.assertEqual(sev(r, "PINNED_MEM_MISJUDGE"), "WARN")
        self.assertEqual(sev(r, "SDPA_BACKEND_FALLBACK"), "WARN")


class TestPipPlan(unittest.TestCase):
    # 今天事故的真实输出格式
    REAL = """
vllm 0.33.0 requires torch==2.14.0, but you have torch 2.15.0 which is incompatible.
vllm 0.33.0 requires transformers<5.21.0,>=5.16.1, but you have transformers 5.23.0 which is incompatible.
numba 0.65.0 requires numpy<2.5,>=1.22, but you have numpy 2.5.4 which is incompatible.
gradio 5.32.1 requires pillow<12.0,>=8.0, but you have pillow 13.0.0 which is incompatible.
"""

    def test_three_critical_detected(self):
        r = sd.parse_pip_plan(self.REAL)
        self.assertEqual(codes(r), {"PIP_BREAKS_CRITICAL"})
        self.assertEqual(len(r), 3)  # torch / transformers / numpy
        self.assertTrue(all(x.severity == "CRITICAL" for x in r))

    def test_non_critical_ignored(self):
        r = sd.parse_pip_plan(
            "gradio 5.32.1 requires pillow<12.0,>=8.0, but you have pillow 13.0.0 which is incompatible."
        )
        self.assertEqual(r, [])

    def test_regression_version_between_pkg_and_which(self):
        # 第一版正则写成 (\S+)\s+which，漏掉中间版本号，静默输出"没有发现"
        r = sd.parse_pip_plan(
            "vllm 0.33.0 requires torch==2.14.0, but you have torch 2.15.0 which is incompatible."
        )
        self.assertEqual(len(r), 1)

    def test_regression_comma_in_constraint(self):
        # 第二版用 [^,;] 截断了带逗号的约束，transformers/numpy 漏检
        r = sd.parse_pip_plan(
            "vllm 0.33.0 requires transformers<5.21.0,>=5.16.1, but you have transformers 5.23.0 which is incompatible."
        )
        self.assertEqual(len(r), 1)

    def test_clean_plan_passes(self):
        r = sd.parse_pip_plan("Successfully installed gptqmodel-7.10.0")
        self.assertEqual(r, [])


if __name__ == "__main__":
    unittest.main()
