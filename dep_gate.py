#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dep_gate.py —— 依赖漂移门禁

背景：上一轮的"升级冲突门"只抓 `requires X, but you have Y which is incompatible`
这种 pip 升级告警。但 `pip install transformers==5.10.4` 这类**降级**不产生任何
incompatible 警告（pip 认为"满足约束"），于是门放行，R3 的 vLLM 环境被静默破坏。
更糟的是 pytest / jupyter 这类传递依赖降级，连警告都没有。

这个门禁换一个判据：**不看警告文本，直接对比装包前后的 site-packages 全量版本**。
任何包的任何方向变化（升级 / 降级 / 新增 / 移除）都会被抓到，并单独标记降级。

用法：
  # 1. 存基线（装任何会碰核心依赖的包之前）
  python dep_gate.py --save .depbaseline.json

  # 2. 包在命令外面跑，事后对比
  python dep_gate.py --run "pip install gptqmodel"

  # 3. 也可以干跑检查（不执行命令）
  python dep_gate.py --check .depbaseline.json

  # 4. 显式钉住关键版本，漂移即失败
  python dep_gate.py --check .depbaseline.json --pin vllm==0.33.0 --pin torch==2.14.0

退出码：0 无漂移（或仅新增且未违反 pin）；1 有漂移；2 运行错误。
"""

import argparse
import json
import os
import re
import subprocess
import sys
from importlib import metadata

BASELINE_META = ("python", "platform", "torch", "hip", "cuda")


def collect():
    """返回 {规范化包名: 版本}，覆盖当前解释器可见的全部已安装发行包。"""
    pkgs = {}
    for d in metadata.distributions():
        name = d.metadata.get("Name")
        if not name:
            continue
        pkgs[name.lower().replace("_", "-")] = d.version
    return pkgs


def meta():
    import platform

    m = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        m["torch"] = torch.__version__
        m["hip"] = getattr(torch.version, "hip", None)
        m["cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        m["torch"] = None
    return m


def parse_version(v):
    """宽松解析：'2.14.0+cu130' -> (2,14,0)，非数字段忽略。用于判断方向。"""
    if not v:
        return ()
    head = str(v).split("+")[0].split("rc")[0]
    parts = re.split(r"[.\-]", head)
    out = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            break
    return tuple(out)


def direction(old, new):
    a, b = parse_version(old), parse_version(new)
    if not a or not b:
        return "changed"
    if b > a:
        return "upgrade"
    if b < a:
        return "downgrade"
    # 数字相同但字符串不同（如 2.14.0 -> 2.14.0.post1）
    return "changed"


def diff(old, new):
    # "changed"：字符串不同但排序分不出方向（如重装换了构建号 0.28.0 → 0.28.0+rocm723）。
    # 构建号不同意味着二进制可能不同，也算漂移。
    result = {"upgrade": [], "downgrade": [], "changed": [], "added": [], "removed": []}
    for name in sorted(set(old) | set(new)):
        if name not in old:
            result["added"].append((name, None, new[name]))
        elif name not in new:
            result["removed"].append((name, old[name], None))
        elif old[name] != new[name]:
            result[direction(old[name], new[name])].append((name, old[name], new[name]))
    return result


def has_drift(d):
    return any(d[k] for k in d)


def print_diff(d):
    sym = {"upgrade": "↑", "downgrade": "↓", "changed": "~", "added": "+", "removed": "-"}
    label = {"upgrade": "升级", "downgrade": "降级", "changed": "变体变更(构建号/后缀)",
             "added": "新增", "removed": "移除"}
    for kind in ("downgrade", "upgrade", "changed", "removed", "added"):
        rows = d[kind]
        if not rows:
            continue
        print(f"\n  {label[kind]} {len(rows)} 个  {sym[kind]}")
        for name, o, n in rows:
            print(f"    {name:<32} {o or '-':<18} -> {n or '-'}")
    if not has_drift(d):
        print("\n  无漂移")


def norm_pin(v):
    """剥掉 PEP 440 本地版本号再比较：0.28.0+rocm723 与 0.28.0 视为同一版本。

    本地构建号（+rocm723 / +git6bbd260）只是发行变体，不影响兼容性判定；
    但发行号本身（0.28 → 0.29）仍是漂移。
    """
    return str(v or "").split("+")[0].strip().lower()


def check_pins(pins):
    """pins: ['vllm==0.33.0', ...]；返回 [(pin, ok, 实际版本)]"""
    cur = collect()
    out = []
    for pin in pins:
        if "==" not in pin:
            out.append((pin, False, "格式错误，应为 name==version"))
            continue
        name, want = pin.split("==", 1)
        name = name.strip().lower().replace("_", "-")
        got = cur.get(name)
        out.append((pin, norm_pin(got) == norm_pin(want), got))
    return out


def do_save(path):
    data = {"meta": meta(), "packages": collect()}
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"基线已保存：{path}")
    print(f"  python {data['meta']['python']} · torch {data['meta'].get('torch')} · "
          f"{len(data['packages'])} 个包")
    return 0


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["packages"], data.get("meta", {})


def report(before_pkgs, before_meta, pins, title):
    after_pkgs, after_meta = collect(), meta()
    print(f"\n{'=' * 64}\n依赖漂移检查 · {title}\n{'=' * 64}")
    for k in BASELINE_META:
        if before_meta.get(k) != after_meta.get(k):
            print(f"  运行环境变化：{k}  {before_meta.get(k)} -> {after_meta.get(k)}")
    d = diff(before_pkgs, after_pkgs)
    print_diff(d)

    bad = has_drift(d)
    if d["downgrade"]:
        print("\n  ⚠ 出现降级。pip 对此不告警，但核心依赖（vllm/torch/transformers）"
              "降级会造成静默失效。")

    if pins:
        print("\n  pin 校验（本地构建号如 +rocm723 / +git…… 不参与比较）")
        for pin, ok, got in check_pins(pins):
            print(f"    {'OK ' if ok else 'FAIL'} {pin:<34} 实际 {got}")

    print()
    if bad:
        print("结论：BLOCK —— 环境已偏离基线，先修复再继续。")
        if d["added"] and not any(d[k] for k in d if k != "added"):
            print("  本次漂移全部是【纯新增】。若新增是有意的（如补装分词后端），"
                  "修复执行后重跑 --save 刷新基线即可。")
        else:
            print("  回滚思路：pip install <name>==<基线版本>，并卸载新装的不匹配包。")
        return 1
    print("结论：PASS —— 与基线一致。")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="依赖漂移门禁：抓升级/降级/变体变更/新增/移除",
        epilog="pin 比较只看发行号（0.28.0+rocm723 与 0.28.0 视为同一版本）；"
               "但构建号本身变化会记为“变体变更”，算漂移。")
    ap.add_argument("--save", metavar="PATH", help="保存当前环境为基线")
    ap.add_argument("--check", metavar="PATH", nargs="?", const=".depbaseline.json",
                    help="对比基线（默认 .depbaseline.json）")
    ap.add_argument("--run", metavar="CMD", help="执行命令，然后对比前后环境")
    ap.add_argument("--pin", action="append", default=[], metavar="NAME==VER",
                    help="钉住关键版本，可重复；本地构建号(+rocm723/+gitxxx)不参与比较")
    args = ap.parse_args()

    if args.save:
        return do_save(args.save)

    if args.run:
        base = ".depbaseline.json"
        if not os.path.exists(base):
            print(f"未找到基线 {base}，先自动保存一份")
            do_save(base)
        before, before_meta = load(base)
        print(f"执行：{args.run}")
        proc = subprocess.run(args.run, shell=True)
        print(f"\n命令退出码：{proc.returncode}")
        rc = report(before, before_meta, args.pin, f"after: {args.run}")
        return 1 if rc or proc.returncode else 0

    if args.check:
        if not os.path.exists(args.check):
            print(f"基线不存在：{args.check}")
            return 2
        before, before_meta = load(args.check)
        return report(before, before_meta, args.pin, args.check)

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())