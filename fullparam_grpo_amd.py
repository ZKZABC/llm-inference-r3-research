#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零依赖全参 GRPO · AMD 192GB（绕开 verl 的版本地狱）

verl 把 transformers 钉在 <5.11，与 R3 vLLM(5.15) 互斥；GSM8K 又要翻墙。
本脚本只用现成 torch(ROCm) + transformers，做最小但真实的全参 GRPO：
  rollout(model.generate) → reward(精确匹配) → group-normalize advantage
  → policy-gradient 更新全部参数。

内存（OLMoE-7B, bf16）：权重 14G + 梯度 14G + AdamW(bf16 states) 28G ≈ 56G + 激活，
192G 绰绰有余，不需要 FSDP / offload。

相对上一版修掉的四个问题：
  1. 填充 token 参与了 logprob 累加 → 用 pad 位置 mask 掉（上一版会往梯度里掺噪声）
  2. 全组同分时 advantage 全 0，loss=0 却照样 step → 现在跳过该组并计数
  3. rollout 在 train 模式下跑（dropout/某些 kernel 行为不同）→ 现在 eval/train 分阶段
  4. transformers 5.x 的 `torch_dtype` 已改名 `dtype` → 两个都兼容

用法：
  python fullparam_grpo_amd.py --selftest          # CPU 机械自检（不需要真实模型）
  python fullparam_grpo_amd.py --model /mnt/workspace/olmoe --steps 2
"""

import argparse
import re
import sys

import torch
import torch.nn.functional as F

QUESTIONS = [
    ("What is 17 + 28?", "45"),
    ("What is 6 * 7?", "42"),
    ("What is 100 - 37?", "63"),
    ("What is 9 * 8?", "72"),
]


def build_tiny():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=8,
        num_experts_per_tok=4,
        head_dim=16,
        max_position_embeddings=256,
        eos_token_id=1,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    return OlmoeForCausalLM(cfg)


def last_number(text):
    m = re.findall(r"-?\d+", text)
    return m[-1] if m else None


def reward(resp, answer):
    return 1.0 if last_number(resp) == answer else 0.0


def sequence_logprobs(model, ids, prompt_len, pad_id=None):
    """返回 (每条序列的 response logprob 之和, 每条序列的有效 response token 数)。

    位置对齐：logits[:, i] 用于预测 ids[:, i+1]，所以 ids[:, i+1] 的 logprob
    落在 lp[:, i]。response 从 ids 的第 prompt_len 个 token 开始，对应 lp 的
    第 prompt_len-1 个位置。pad / eos 之后的填充位必须 mask 掉。
    """
    logits = model(ids).logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

    mask = torch.zeros_like(lp, dtype=torch.bool)
    mask[:, prompt_len - 1:] = True
    if pad_id is not None:
        mask &= tgt != pad_id

    return (lp * mask).sum(-1), mask.sum(-1)


def group_advantage(rewards):
    """GRPO 组内标准化。全组同分时返回全 0（并让调用方跳过）。"""
    r = torch.as_tensor(rewards, dtype=torch.float32)
    if r.numel() == 0:
        return r
    if float(r.std()) < 1e-8:
        return torch.zeros_like(r)
    return (r - r.mean()) / (r.std() + 1e-8)


def tokenizer_diag(path):
    """tokenizer 装载失败时的自诊断：目录里到底是什么、后端缺哪个。"""
    import os

    lines = []
    try:
        files = sorted(os.listdir(path))
    except Exception as e:  # noqa: BLE001
        return [f"  目录不可读：{path} -> {e}"]
    keys = ("token", "vocab", "merge", "special_tokens", "added", "spiece", "json")
    shown = [f for f in files if any(k in f.lower() for k in keys)][:14]
    lines.append(f"  目录 {path}")
    lines.append(f"  相关文件：{', '.join(shown) or '（一个分词相关文件都没有）'}")
    has_fast = "tokenizer.json" in files
    lines.append(f"  tokenizer.json：{'存在' if has_fast else '不存在 → 必须走慢速分词器/外部后端路线'}")
    for mod in ("sentencepiece", "tiktoken"):
        try:
            __import__(mod)
            lines.append(f"  {mod}：已安装")
        except ImportError:
            lines.append(f"  {mod}：未安装；若上面报的就是它 → pip install {mod}")
    lines.append("  若目录缺分词文件：用 --tokenizer 指向完整分词器目录，或从 modelscope 重下分词器。")
    return lines


def load_model(path, tok_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_path = tok_path or path
    try:
        tok = AutoTokenizer.from_pretrained(tok_path)
    except Exception as e:  # noqa: BLE001
        print("=" * 68)
        print("tokenizer 加载失败 —— 原因与下一步：")
        print(f"  {type(e).__name__}: {e}")
        for line in tokenizer_diag(tok_path):
            print(line)
        print("=" * 68)
        raise SystemExit(1) from e

    kwargs = {"trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kwargs)
    except TypeError:  # 旧版 transformers 只认 torch_dtype
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, **kwargs)
    return model, tok


def rollout(model, prompt, group, max_new, temperature, top_p, pad_id):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        gen = model.generate(
            prompt,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group,
            pad_token_id=pad_id,
        )
    if was_training:
        model.train()
    return gen


def build_loss(model, gen, prompt_len, adv, pad_id):
    """token 级策略梯度：sum(adv_tok * logp) / 有效 token 数。"""
    logits = model(gen).logits[:, :-1, :].float()
    tgt = gen[:, 1:]
    lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    mask = torch.zeros_like(lp, dtype=torch.bool)
    mask[:, prompt_len - 1:] = True
    if pad_id is not None:
        mask &= tgt != pad_id

    adv_tok = adv.unsqueeze(1).expand_as(lp)
    n = mask.sum().clamp(min=1)
    return -(adv_tok * lp * mask).sum() / n


def selftest():
    """机械自检：不依赖真实模型，验证形状 / mask / 梯度 / 参数更新 / 退化组。"""
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok = ok and bool(cond)

    print("=" * 68)
    print("全参 GRPO 机械自检")
    print("=" * 68)

    check("reward 解析 '17+28=45'", reward("45", "45") == 1.0 and reward("44", "45") == 0.0)
    check("last_number 取最后一个", last_number("step 1 then 42") == "42")

    adv = group_advantage([1.0, 0.0, 1.0, 0.0])
    check("组内标准化 adv 对称", abs(float(adv.mean())) < 1e-6 and float(adv.std()) > 0.99,
          f"adv={adv.tolist()}")
    deg = group_advantage([1.0, 1.0, 1.0, 1.0])
    check("全同分组 adv 全 0", float(deg.abs().max()) == 0.0)

    model = build_tiny()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    pad_id = 0
    G = 4
    prompt = torch.randint(1, 64, (1, 6))

    gen = rollout(model, prompt, G, 8, 1.0, 1.0, pad_id)
    check("rollout 组大小", gen.shape[0] == G, f"shape={tuple(gen.shape)}")
    check("prompt 前缀保留", torch.equal(gen[:, :6], prompt.expand(G, -1)))

    # 用与生成内容相关的合成奖励，保证组内有方差（真实链路用文本精确匹配）
    rewards = [(float(gen[g, 6:].float().mean()) % 1.0) for g in range(G)]
    adv = group_advantage(rewards)
    check("合成奖励产生非零 adv", float(adv.abs().max()) > 0, f"adv={[round(x,3) for x in adv.tolist()]}")

    before = model.model.layers[0].mlp.gate.weight.detach().clone()
    loss = build_loss(model, gen, 6, adv, pad_id)
    check("loss 有限", bool(torch.isfinite(loss)), f"loss={loss.item():.4f}")
    opt.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    check("梯度范数有限且非零", bool(torch.isfinite(gnorm)) and float(gnorm) > 0,
          f"|g|={float(gnorm):.3e}")
    opt.step()
    after = model.model.layers[0].mlp.gate.weight.detach()
    check("参数确实更新", not torch.equal(before, after),
          f"Δmax={float((before - after).abs().max()):.3e}")

    # pad mask：在 response 尾部追加 pad，logprob 不应变化
    lp1, n1 = sequence_logprobs(model, gen, 6, pad_id)
    pad = torch.full((G, 3), pad_id, dtype=gen.dtype)
    lp2, n2 = sequence_logprobs(model, torch.cat([gen, pad], 1), 6, pad_id)
    check("pad 位置被 mask（有效 token 数不变）", torch.equal(n1, n2),
          f"{int(n1[0])} vs {int(n2[0])}")

    print("-" * 68)
    print("结论：", "全部通过，链路可用" if ok else "存在失败项，先修再上大模型")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="本地模型目录（含权重与分词器）")
    ap.add_argument("--tokenizer", default=None,
                    help="分词器目录，默认与 --model 一致；模型目录缺分词文件时指向别处")
    ap.add_argument("--selftest", action="store_true", help="CPU 机械自检")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-prompt", type=int, default=256)
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if not args.model:
        ap.error("需要 --model，或使用 --selftest")

    model, tok = load_model(args.model, args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    n_param = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"device={device} params={n_param:.2f}B group={args.group} lr={args.lr}")

    for step in range(args.steps):
        losses, mean_r, skipped = [], [], 0
        for q, ans in QUESTIONS:
            ids = tok(q + " Answer:", return_tensors="pt").input_ids[:, -args.max_prompt:]
            ids = ids.to(device)

            gen = rollout(model, ids, args.group, args.max_new,
                          args.temperature, args.top_p, pad_id)
            prompt_len = ids.shape[1]
            rewards = [reward(tok.decode(gen[g, prompt_len:], skip_special_tokens=True), ans)
                       for g in range(args.group)]
            adv = group_advantage(rewards).to(device)
            mean_r.append(sum(rewards) / len(rewards))
            if float(adv.abs().max()) == 0.0:
                skipped += 1
                continue

            model.train()
            loss = build_loss(model, gen, prompt_len, adv, pad_id)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            losses.append(loss.item())

        lr_ = sum(losses) / len(losses) if losses else 0.0
        mem = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
        print(f"step {step}: loss={lr_:.4f} mean_reward={sum(mean_r)/len(mean_r):.2f} "
              f"skipped={skipped}/{len(QUESTIONS)} peak={mem:.1f}G")

    print("全参 GRPO 流程闭环（rollout→reward→advantage→全参更新）")
    return 0


if __name__ == "__main__":
    sys.exit(main())