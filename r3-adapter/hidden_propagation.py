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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8000")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--model-dir", default="/mnt/workspace/model")
    ap.add_argument("--model-name", default="")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args()

    host = f"http://localhost:{args.port}" if args.port else args.host

    try:
        from gptqmodel import GPTQModel
    except ImportError as e:
        print(f"需要 gptqmodel: {e}")
        return 2

    model_name = resolve_model(host, args.model_name)
    print("=" * 70)
    print("误差传播曲线")
    print("=" * 70)
    print(f"\n  prompt   {args.prompt!r}")
    print(f"  vLLM     {host}")

    print(f"\n加载模型（约 1 分钟）...")
    gm = GPTQModel.load(args.model_dir, device="cpu")
    tm = gm.model.eval()
    tok = getattr(gm, "tokenizer", None)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    ids = tok(args.prompt, return_tensors="pt")["input_ids"]
    k = int(block_attr(tm.model.layers[0].mlp, "top_k", 4))
    print(f"  完成：{len(tm.model.layers)} 层，{len(ids[0])} token，top_k={k}")

    rv = vllm_routes(host, model_name, args.prompt)
    print(f"\n  vLLM 路由 shape={rv.shape}")

    n_tok = len(ids[0])
    if rv.shape[0] != n_tok:
        print(f"\n  [警告] vLLM 路由 {rv.shape[0]} 行 != token {n_tok} 个")
        n = min(rv.shape[0], n_tok)
        rv = rv[:n]
        ids = ids[:, :n]
        print(f"  截断到 {n} 个 token 后继续")

    print("\n跑两次前向（原始 / 注入 vLLM 路由）")
    logits_o, h_o = run_with_hidden(tm, ids, None)
    logits_i, h_i = run_with_hidden(tm, ids, routes=rv)

    # 自检：per_layer[0] 到底是 layer0 的输出，还是 embedding？
    # 实测发现 L0 差异恒为 0，两个不同引擎不可能 bit-identical，
    # 所以更可能是 hook 抓到了 embedding。
    with torch.no_grad():
        emb = tm.model.embed_tokens(ids).float()
    is_emb = h_o[0].shape == emb.shape and torch.equal(h_o[0], emb)
    if is_emb:
        print("\n  [注意] per_layer[0] 等于 embedding —— 层号需整体减 1")
        print("         register_forward_hook 抓到的是该层的输入，不是输出")
    else:
        print("\n  [自检] per_layer[0] 不是 embedding，层号按原样理解")

    print("\n逐层 hidden 差异")
    print("─" * 70)
    print(f"  {'层':<5} {'max|Δh|':>12} {'max|h|':>12} {'相对':>10}  趋势")
    print("  " + "-" * 62)

    rel_list = []
    for i in range(len(h_o)):
        d = (h_o[i] - h_i[i]).abs().max().item()
        scale = h_o[i].abs().max().item()
        rel = d / max(scale, 1e-9)
        rel_list.append(rel)
        bar = "█" * min(30, int(rel * 300))
        label = f"L{i}" if not is_emb else f"L{i - 1}*"
        print(f"  {label:<5} {d:>12.3e} {scale:>12.3e} {rel:>9.2e}  {bar}")

    d_logits = (logits_o - logits_i).abs().max().item()
    print(f"\n  最终 logits 差异  {d_logits:.3e}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)

    n = len(rel_list)
    first_q = float(np.mean(rel_list[: n // 4]))
    last_q = float(np.mean(rel_list[-n // 4 :]))
    ratio = last_q / max(first_q, 1e-12)
    print(f"  前 1/4 层平均相对差异  {first_q:.3e}")
    print(f"  后 1/4 层平均相对差异  {last_q:.3e}")
    print(f"  比值                   {ratio:.2f}x")

    if ratio > 2.0:
        print("\n  → 单调上升，符合「误差累积」假设 A")
        print("    浅层路由差异被放大到深层。")
        print("    含义：R3 必须全层修，少修一层后面全被污染。")
    elif ratio < 1.3:
        print("\n  → 基本持平，符合「深层 router 更敏感」假设 B")
        print("    hidden 差异没有放大，但同样的差异在深层引起更多翻转。")
        print("    含义：可以对深层单独加精度，浅层可放宽。")
    else:
        print("\n  → 介于两者之间，两个因素都有贡献。")

    print("\n  注意：这个实验只用了 1 条 prompt，趋势需多样本确认。")
    if is_emb:
        print("  另外：层号已按 embedding 自检结果标注，带 * 的行是调整后的。")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
