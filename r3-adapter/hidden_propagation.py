#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
误差传播曲线：注入 vLLM 路由后，hidden state 的差异怎么随层变化

报告 §6.1 发现「深层一致率更低」（L0–L7 均 92.99%，L20–L23 均 87.78%）。
有两种解释，这个脚本用来区分：

  A. 误差累积（正反馈）
     浅层路由差 → hidden 不同 → 深层 router 输入不同 → 深层路由差更多
     预期：hidden 差异随层数单调上升

  B. 深层 router 更敏感
     hidden 差异基本持平，但同样的 hidden 差异在深层引起更多 top-k 翻转
     预期：hidden 差异持平或下降，但路由差异上升

两者含义完全不同：
  A → R3 必须全层修，少修一层后面全被污染
  B → 深层值得特殊处理（比如只对深层用更高精度的 router）

用法:
    python hidden_propagation.py --port 8000
    python hidden_propagation.py --port 8000 --prompt "The capital of France is"
"""

import argparse
import base64
import io
import json
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _post(url, payload, timeout=300):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model(host, given):
    if given:
        return given
    try:
        req = urllib.request.Request(f"{host}/v1/models")
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read().decode("utf-8")).get("data", [])
        if items:
            return items[0]["id"]
    except Exception:
        pass
    return "default"


def vllm_routes(host, model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "repetition_penalty": 1.0,
    }
    data = _post(f"{host}/v1/completions", payload)
    raw = data["choices"][0].get("routed_experts")
    if raw is None:
        raise RuntimeError("响应里没有 routed_experts")
    return np.load(io.BytesIO(base64.b64decode(raw)))


def gate_logits(block, h):
    res = block.gate(h)
    return res[0] if isinstance(res, tuple) else res


def block_attr(block, name, default=None):
    if hasattr(block, name):
        return getattr(block, name)
    if hasattr(block.gate, name):
        return getattr(block.gate, name)
    return default


class InjectedMoE(nn.Module):
    def __init__(self, hf_block, route_ids):
        super().__init__()
        self.gate = hf_block.gate
        self.experts = hf_block.experts
        self.shared_expert = getattr(hf_block, "shared_expert", None)
        self.shared_expert_gate = getattr(hf_block, "shared_expert_gate", None)
        self.route_ids = route_ids
        self.norm_topk_prob = bool(block_attr(hf_block, "norm_topk_prob", False))

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        h = hidden_states.view(-1, H)
        if self.route_ids.shape[0] != h.shape[0]:
            raise ValueError(
                f"路由 {self.route_ids.shape[0]} 行 != token {h.shape[0]} 个"
            )

        logits = gate_logits(self, h)
        probs = F.softmax(logits, dim=-1, dtype=torch.float)
        w = probs.gather(-1, self.route_ids)
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        w = w.to(logits.dtype)

        n_exp = int(self.route_ids.max().item()) + 1
        final = torch.zeros_like(h)
        for e in range(n_exp):
            mask = self.route_ids == e
            if not mask.any():
                continue
            pos, slot = torch.where(mask)
            if hasattr(self.experts, "gate_up_proj"):
                gu = F.linear(h[pos], self.experts.gate_up_proj[e])
                gate, up = gu.chunk(2, dim=-1)
                y = F.linear(self.experts.act_fn(gate) * up, self.experts.down_proj[e])
            else:
                y = self.experts[e](h[pos])
            y = y * w[pos, slot].unsqueeze(-1)
            final.index_add_(0, pos, y.to(final.dtype))

        if self.shared_expert is not None:
            sh = self.shared_expert(h)
            sh = torch.sigmoid(self.shared_expert_gate(h)) * sh
            final = final + sh

        return final.view(B, S, H)


def run_with_hidden(model, ids, routes):
    """返回 (logits, 每层输出的 hidden list)；routes=None 表示不注入"""
    per_layer = []
    hooks = []
    saved = []

    if routes is not None:
        for i, layer in enumerate(model.model.layers):
            saved.append((layer, layer.mlp))
            r = torch.as_tensor(routes[:, i, :], dtype=torch.long)
            layer.mlp = InjectedMoE(layer.mlp, r)

    for layer in model.model.layers:
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            per_layer.append(h.detach().float())
        hooks.append(layer.register_forward_hook(hook))

    try:
        with torch.no_grad():
            out = model(ids)
    finally:
        for hd in hooks:
            hd.remove()
        for layer, orig in saved:
            layer.mlp = orig

    return out.logits, per_layer


PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
    "请解释一下量子纠缠的基本原理",
    "SELECT u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id",
    "The patient presented with acute abdominal pain and fever. The differential diagnosis includes",
    "证明：对于任意正整数 n，n^3 - n 能被 6 整除。证明过程如下：",
    "写一首关于秋天的五言绝句，要求押平声韵：",
    "The transformer architecture relies on self-attention mechanisms that compute",
    "将下面这段话翻译成英文：人工智能正在改变我们的生活方式",
    "Revenue recognition under ASC 606 requires the entity to identify",
    "机器学习模型的泛化能力取决于训练数据的分布是否能够代表真实场景",
    "fn main() { let mut v: Vec<i32> = Vec::new(); v.push(1); println!(\"{:?}\", v); }",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model-dir", default="/mnt/workspace/model")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--prompt", default=None, help="只跑这一条；不给则跑内置 12 条")
    ap.add_argument("--prompts-file", default=None)
    args = ap.parse_args()

    host = f"http://localhost:{args.port}" if args.port else args.host

    try:
        from gptqmodel import GPTQModel
    except ImportError as e:
        print(f"需要 gptqmodel: {e}")
        return 2

    prompts = [args.prompt] if args.prompt else PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    model_name = resolve_model(host, args.model_name)
    print("=" * 70)
    print("误差传播曲线 · 多样本")
    print("=" * 70)
    print(f"\n  vLLM     {host}")
    print(f"  prompts  {len(prompts)}")

    print("\n加载模型（约 1 分钟）...")
    gm = GPTQModel.load(args.model_dir, device="cpu")
    tm = gm.model.eval()
    tok = getattr(gm, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    k = int(block_attr(tm.model.layers[0].mlp, "top_k", 4))
    print(f"  完成：{len(tm.model.layers)} 层，top_k={k}")

    curves, ratios, tok_counts = [], [], []
    abs_curves = []
    is_emb = None

    print("\n逐 prompt")
    print("─" * 70)
    for idx, p in enumerate(prompts):
        tag = p[:34].replace("\n", " ")
        try:
            rv = vllm_routes(host, model_name, p)
            ids = tok(p, return_tensors="pt")["input_ids"]
            n_tok = len(ids[0])

            if rv.shape[0] != n_tok:
                n = min(rv.shape[0], n_tok)
                rv = rv[:n]
                ids = ids[:, :n]

            _, h_o = run_with_hidden(tm, ids, None)
            _, h_i = run_with_hidden(tm, ids, rv)

            if is_emb is None:
                with torch.no_grad():
                    emb = tm.model.embed_tokens(ids).float()
                is_emb = h_o[0].shape == emb.shape and torch.equal(h_o[0], emb)

            rel = []
            abs_diff = []
            for i in range(len(h_o)):
                d = (h_o[i] - h_i[i]).abs().max().item()
                s = h_o[i].abs().max().item()
                abs_diff.append(d)
                rel.append(d / max(s, 1e-9))

            curves.append(rel)
            abs_curves.append(abs_diff)
            # 前 1/4 接近 0 时比值会爆炸（实测出现过 1.1e10x），加下限保护
            q = max(1, len(rel) // 4)
            base = float(np.mean(rel[:q]))
            r = float(np.mean(rel[-q:]) / base) if base > 1e-10 else float("nan")
            ratios.append(r)
            tok_counts.append(n_tok)
            rs_txt = f"{r:>7.1f}x" if np.isfinite(r) else "    n/a"
            print(f"  [{idx + 1:>2}/{len(prompts)}] {n_tok:>3} tok  ratio={rs_txt}  {tag}")
        except Exception as e:
            print(f"  [{idx + 1:>2}/{len(prompts)}] 失败 {type(e).__name__}: {str(e)[:44]}")

    if not curves:
        print("\n没有成功的数据。")
        return 1

    C = np.array(curves)
    A = np.array(abs_curves)
    mean_curve = C.mean(axis=0)
    med_curve = np.median(C, axis=0)
    mean_abs = A.mean(axis=0)

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"  成功 prompt   {len(curves)}/{len(prompts)}")
    print(f"  总 token      {sum(tok_counts)}")

    if is_emb:
        print("\n  [注意] per_layer[0] 等于 embedding —— 层号需整体减 1")
    else:
        print("\n  [自检] per_layer[0] 不是 embedding，层号按原样理解")

    rs = np.array([r for r in ratios if np.isfinite(r)])
    print(f"\n  各 prompt 的后/前 1/4 比值（{len(rs)}/{len(ratios)} 条有效）：")
    if len(rs):
        print(f"    中位数 {np.median(rs):.1f}x   最小 {rs.min():.1f}x   最大 {rs.max():.1f}x")
        print(f"    均值 {rs.mean():.1f}x ± {rs.std():.1f}x")
        if len(rs) < len(ratios):
            print(f"    注：{len(ratios) - len(rs)} 条因前 1/4 差异过小无法计算，已排除")

    print(f"\n  逐层差异（{len(mean_curve)} 层）：")
    print(f"  {'层':<5} {'相对(均值)':>12} {'相对(中位)':>12} {'绝对(均值)':>12}  趋势")
    print("  " + "-" * 62)
    peak = max(float(mean_curve.max()), 1e-12)
    for i, v in enumerate(mean_curve):
        bar = "█" * min(30, int(v / peak * 30))
        label = f"L{i}" if not is_emb else f"L{i - 1}*"
        print(f"  {label:<5} {v:>12.3e} {med_curve[i]:>12.3e} {mean_abs[i]:>12.3e}  {bar}")

    # L0 的相对差异受分母影响（hidden 量级只有 L1 的 1/25 左右），
    # 跨层比较时把 L0 排除，否则会低估放大倍数。
    start = 1 if len(mean_curve) > 4 else 0
    core = mean_curve[start:]
    n = len(core)
    q = max(1, n // 4)
    first = float(np.mean(core[:q]))
    last = float(np.mean(core[-q:]))
    overall = last / max(first, 1e-12)

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"  统计区间       L{start} – L{len(mean_curve) - 1}（排除 L0，其分母量级异常）")
    print(f"  前 1/4 层平均  {first:.3e}")
    print(f"  后 1/4 层平均  {last:.3e}")
    print(f"  整体比值       {overall:.1f}x（单条中位数 {np.median(rs) if len(rs) else float('nan'):.1f}x）")

    if overall > 2.0:
        print("\n  → 单调上升，符合「误差累积」假设 A")
        print("    含义：R3 必须全层修，少修一层后面全被污染。")
    elif overall < 1.3:
        print("\n  → 基本持平，符合「深层 router 更敏感」假设 B")
        print("    含义：深层可单独加精度，浅层可放宽。")
    else:
        print("\n  → 介于两者之间，两个因素都有贡献。")

    n_up = int((rs > 2.0).sum()) if len(rs) else 0
    print(f"\n  稳健性：{n_up}/{len(rs)} 条 prompt 单独看也 >2x")
    if len(rs) and n_up >= len(rs) * 0.7:
        print("  → 趋势在绝大多数样本上一致，可信")
    elif len(rs):
        print("  → 样本间差异较大，结论需更多数据")

    # 找放大最剧烈的几层（相对上一层的跳变）
    jumps = []
    for i in range(start + 1, len(mean_curve)):
        prev = mean_curve[i - 1]
        if prev > 1e-12:
            jumps.append((i, mean_curve[i] / prev))
    if jumps:
        jumps.sort(key=lambda x: -x[1])
        top = [(i, r) for i, r in jumps[:3] if r > 1.5]
        if top:
            print(f"\n  放大最剧烈的层（相对上一层）：")
            for i, r in top:
                print(f"    L{i - 1} → L{i}   {r:.1f}x")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
