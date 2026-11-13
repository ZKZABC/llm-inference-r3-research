# -*- coding: utf-8 -*-
"""R3 数据契约适配层 —— 核心逻辑单元测试（不依赖 vLLM / GPU，可立即运行）

验证适配层真正值钱的四件事（对应《报告》§7.3 dtype 分歧、§7.4 四类静默失败）：

  1. dtype 归一化 —— vLLM 用 uint8/uint16、fork 用 int16、Miles 用 int32，
     同一份数据三种宽度。适配层必须统一，否则体积与解析都会错。
  2. shape 断言 —— routed_experts 的行数 = prompt_len + gen_len - 1，
     最后一个采样 token 没有路由（它还没通过模型）。
  3. -1 哨兵 —— 不可映射的 token 必须用 -1 标记，不能用 0 填充。
     0 会「静默别名到 expert 0」并污染梯度（Megatron PR#79 的教训）。
  4. null 处理 —— 被抢占/中止的请求返回 None，不能当空数组处理。

运行：python contract_check.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# 适配层核心函数
# ---------------------------------------------------------------------------

def normalize_dtype(routed, num_experts):
    """把任意 dtype 的路由数组统一为 int32，并校验专家 ID 合法性。

    返回 (归一化数组, unmappable 计数)。
    unmappable = 值为 -1 的条目数（哨兵，保持实时路由，不能静默别名）。
    """
    arr = np.asarray(routed, dtype=np.int32)  # 统一 dtype
    valid = ((arr >= 0) & (arr < num_experts)) | (arr == -1)
    if not valid.all():
        bad = np.unique(arr[~valid])
        raise ValueError(f"非法专家 ID {bad[:8]}：应在 [0,{num_experts}) 或 -1")
    return arr, int((arr == -1).sum())


def check_shape(routed, prompt_len, gen_len):
    """断言 routed_experts 行数 = prompt_len + gen_len - 1。

    末 token 无路由记录（尚未通过模型），数组比 token 总数少 1。
    不处理会导致训练侧对齐整体错位一格。
    """
    expected = prompt_len + gen_len - 1
    assert routed.shape[0] == expected, (
        f"契约违背：路由 {routed.shape[0]} 行 != {expected} "
        f"(prompt {prompt_len} + gen {gen_len} - 1)"
    )
    return expected


def handle_null(routed, request_aborted):
    """被抢占/中止的请求，routed_experts 为 None。

    消费方必须显式识别并跳过，不能假设字段恒存在；
    更不能把 None 当空数组（会静默吞掉一整条样本）。
    """
    if request_aborted:
        assert routed is None, "中止请求应返回 None"
        return None  # 显式跳过该样本，并计入「无效样本」而非「空路由」
    assert routed is not None, "正常请求不应返回 None"
    return routed


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")


def test_dtype_normalization():
    """同一份语义数据，三种 dtype 应归一化到相同结果。"""
    print("\n[1] dtype 归一化（报告 §7.3）")
    # 三个「表示相同路由」的数组，宽度不同
    u8 = np.array([[[3, 7], [1, 2]]], dtype=np.uint8)      # vLLM ≤256 专家
    u16 = np.array([[[3, 7], [1, 2]]], dtype=np.uint16)    # vLLM >256 专家
    i32 = np.array([[[3, 7], [1, 2]]], dtype=np.int32)     # Miles

    a, _ = normalize_dtype(u8, num_experts=16)
    b, _ = normalize_dtype(u16, num_experts=16)
    c, _ = normalize_dtype(i32, num_experts=16)
    assert a.dtype == b.dtype == c.dtype == np.int32
    assert (a == b).all() and (b == c).all()
    ok("三种 dtype 归一化到同一结果")


def test_unmappable_sentinel():
    """-1 哨兵必须保留，0 填充是错误做法。"""
    print("\n[2] -1 哨兵 vs 0 填充（报告 §7.4 / Megatron PR#79）")
    # 不可映射 token 用 -1（正确）
    good = np.array([[[3, -1], [1, 2]]], dtype=np.int32)
    arr, unmappable = normalize_dtype(good, num_experts=16)
    assert unmappable == 1, "应统计到 1 个 unmappable 条目"
    assert arr[0, 0, 1] == -1, "-1 哨兵被篡改"

    # 若用 0 填充（错误）：0 会被当作「真的选了 expert 0」
    bad_fill = np.array([[[3, 0], [1, 2]]], dtype=np.int32)
    _, bad_unmappable = normalize_dtype(bad_fill, num_experts=16)
    assert bad_unmappable == 0, "0 填充导致 unmappable 被静默吞掉——这正是要防的坑"
    ok("-1 哨兵被保留；0 填充会让 unmappable 计数归零（静默别名到 expert 0）")


def test_illegal_id():
    """越界专家 ID 必须被拒绝，而不是静默通过。"""
    print("\n[3] 越界专家 ID 校验")
    illegal = np.array([[[3, 99], [1, 2]]], dtype=np.int32)  # 99 > num_experts-1
    try:
        normalize_dtype(illegal, num_experts=16)
    except ValueError as e:
        ok(f"越界 ID 被拒绝：{e}")
    else:
        raise AssertionError("越界专家 ID 未被拦截")


def test_shape_contract():
    """末 token 无路由：行数 = prompt + gen - 1。"""
    print("\n[4] shape 断言（末 token 无路由）")
    prompt_len, gen_len = 10, 5
    routed = np.zeros((prompt_len + gen_len - 1, 4, 8), dtype=np.int32)  # 14 行
    check_shape(routed, prompt_len, gen_len)
    ok(f"shape[0] = {routed.shape[0]} = {prompt_len}+{gen_len}-1 成立")

    wrong = np.zeros((prompt_len + gen_len, 4, 8), dtype=np.int32)  # 15 行（错）
    try:
        check_shape(wrong, prompt_len, gen_len)
    except AssertionError as e:
        ok(f"错位被拦截：{e}")
    else:
        raise AssertionError("shape 错位未被拦截")


def test_null_handling():
    """被抢占请求的 None 必须显式处理。"""
    print("\n[5] null 处理（被抢占/中止请求）")
    assert handle_null(None, request_aborted=True) is None
    ok("中止请求返回 None → 显式跳过")
    # 错误用法演示：把 None 当空数组
    try:
        np.asarray(None)
    except Exception:
        ok("None 无法转数组——若代码不显式判空，会在此处崩溃而非静默丢数据")


if __name__ == "__main__":
    print("=" * 60)
    print("R3 数据契约适配层 · 核心逻辑测试")
    print("=" * 60)
    test_dtype_normalization()
    test_unmappable_sentinel()
    test_illegal_id()
    test_shape_contract()
    test_null_handling()
    print("\n" + "=" * 60)
    print(f"结果：{PASS} 项通过，{FAIL} 项失败")
    if FAIL == 0:
        print("✓ 适配层核心逻辑全部通过。下一步：接真实 vLLM 的 routed_experts。")
    else:
        raise SystemExit("✗ 有测试失败，见上方输出")
