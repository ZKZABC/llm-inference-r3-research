# WSL2 环境搭建 + R3 真实实验清单

> 目标：在你的 Windows（主系统不变）+ WSL2 + 8GB 显卡上，跑通 `r3_verify.py`，
> 拿到真实 MoE 模型的 `routed_experts`，然后开始「挖坑」。
>
> 全程主系统都是 Windows，WSL2 只是 Windows 里的一个 Ubuntu 子系统。

---

## 第 0 步：前置检查（你应该已经满足）

- Windows 10 2004+ 或 Win11
- NVIDIA 驱动已装（你在 Windows 里 `nvidia-smi` 能看到 RTX 5060，即已满足）

---

## 第 1 步：安装 WSL2 到 D 盘（一次性，约 10 分钟）

右键「开始」菜单 → **终端(管理员)**，运行：

```powershell
wsl --install -d Ubuntu --location D:\WSL
```

- `--location D:\WSL` 直接把 Ubuntu 的虚拟磁盘装到 D 盘（不占 C 盘）。
- 会自动安装 WSL2 + Ubuntu，然后提示**重启电脑**。
- 重启后会自动弹出 Ubuntu 窗口，让你设置一个 Linux 用户名和密码（记牢，后面要用 sudo）。

> ⚠️ 路径规则：纯英文、无空格、无中文（`D:\WSL` 这种）。`D:\我的WSL` 会报错。

**如果 `--location` 报错（WSL 版本偏旧）→ 用迁移法兜底：**

```powershell
# 1. 先装到 C 盘
wsl --install -d Ubuntu
# （设置好用户名密码后）
# 2. 关闭 WSL，导出
wsl --shutdown
wsl --export Ubuntu D:\WSL\ubuntu-backup.tar
# 3. 注销 C 盘的实例（释放空间）
wsl --unregister Ubuntu
# 4. 导入到 D 盘
wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu-backup.tar --version 2
# 5. 恢复默认登录用户（把 yourname 换成你设的用户名）
ubuntu config --default-user yourname
```

**其它可能报错 → 解法：**

| 报错 | 解法 |
|---|---|
| 提示「需要更新 WSL 内核」 | `wsl --update` 后重试 |
| 提示「虚拟机平台未启用」 | `dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart` 然后重启 |
| 下载 Ubuntu 很慢 | 等，或换网络；不急 |

---

## 第 2 步：确认 WSL2 能直接看到你的显卡

在 Ubuntu 终端里运行：

```bash
nvidia-smi
```

- 能看到 RTX 5060 就成功。**WSL2 直通 GPU，不需要在 Linux 里再装驱动**（驱动用 Windows 侧那份）。

---

## 第 3 步：装 Python + vLLM（Linux 官方支持，编译好的 wheel）

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv build-essential
python3 -m venv ~/r3env
source ~/r3env/bin/activate
pip install vllm
```

**可能报错 → 解法：**

| 报错 | 解法 |
|---|---|
| pip 下载太慢 | 换清华源：`pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple` |
| 提示缺 gcc/g++ | 上面已 `install build-essential`，若还缺就 `sudo apt install -y gcc g++` |

> 每次新开终端进 WSL 后，都要先 `source ~/r3env/bin/activate` 激活环境。

---

## 第 4 步：下载模型（4bit 量化 MoE，约 8GB）

> 实际采用：**`Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`**（AWQ 版已下架 404，GPTQ-Int4 在 vLLM 里支持最好）。

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4', \
local_dir='~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4')"
```

- **为什么用 4bit**：原始 BF16 是 28GB，8GB 卡塞不下；4bit 约 8GB，仍需专家 offload（见第 7 步）。
- **断点续传**：下载中断了没关系，重新跑同一条命令会跳过已下完的文件、对没下完的从断点继续。**千万别删 `~/.cache/huggingface/` 缓存目录**——删了才要重头下。用 `ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4/snapshots/*/` 看进度。
- **先快后慢 → 换国内镜像**（直连国际链路丢包、CDN 限速导致，不是你的问题）：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com   # 国内首选，实测 ~1.85MB/s
  # 备用镜像：https://aifasthub.com（实测 ~2.87MB/s）
  ```
  换镜像后已下完的文件照样复用（sha256 一致），只有正在下的那一个文件可能要重下。若还想更快且网络稳定，可 `pip install -U hf_transfer` + `export HF_HUB_ENABLE_HF_TRANSFER=1` 开多线程（但**不支持断点续传**，网络不稳别开）。
- **可能报错**：`401` 或「需要同意模型协议」→ 去 huggingface.co 注册账号，打开该模型页点「Agree and access repository」，然后在 WSL 里 `huggingface-cli login` 粘贴你的 token。

> ⚠️ 如果 4bit 都塞不下（下面第 6 步会 OOM），那「显存不够」就是你的**第一个真实坑**——记下来，它本身有价值。

---

## 第 5 步：跑验证脚本

把 `r3_verify.py` 拷进 WSL（在 Windows 里文件位于 `C:\Users\lenovo\WorkBuddy\...\r3-adapter\`，WSL 里对应 `/mnt/c/Users/lenovo/WorkBuddy/...`），或直接：

```bash
cd /mnt/c/Users/lenovo/WorkBuddy/2026-11-13-16-48-36/r3-adapter
python r3_verify.py --model ~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 --max-model-len 1024
```

- 脚本会自动开 `--enable-return-routed-experts`，跑一次推理，校验 `routed_experts` 的 shape / dtype / 合法性。
- **大概率 OOM → 按第 7 步加 offload 参数。**

---

## 第 6 步：挖坑 —— 记下你遇到的每一个异常

下面这些是「大概率会撞见」的坑，撞见就记下来，它们就是你的护城河素材：

| 症状 | 可能原因 | 先试这个 |
|---|---|---|
| `CUDA out of memory` | 8GB 塞不下 8GB 权重 | **见第 7 步 OOM 预案**：`--cpu-offload-gb 6 --cpu-offload-params experts --offload-backend uva` |
| `pydantic ValidationError: cpu_offload_params Input should be a valid set` | **vLLM 0.33 的 Python API 要 `set[str]`，不是 `str`**（CLI 会自动转，`LLM(...)` 不会） | 传 `cpu_offload_params={'experts'}`，脚本已自动转换 |
| `Pinned memory is not available on this platform; falling back to device memory` | **WSL2 不支持 pinned memory**，UVA offload 静默失效（权重又回到显存），表现为 `Available KV cache memory: -0.46 GiB` | 换后端：`--offload-backend prefetch --offload-group-size 4 --offload-num-in-group 1`（不传 `--cpu-offload-gb/--cpu-offload-params`） |
| `Value error, offload_num_in_group (1) must be <= offload_group_size (0)` | prefetch 后端必须显式给 group_size | 加 `--offload-group-size 4`（0 表示禁用） |
| `ValueError: Free memory on device cuda:0 (x/y GiB) on startup is less than desired GPU memory utilization` | `gpu_memory_utilization` 校验的是**启动时空闲显存**，不是总显存；别的程序占了显存就到不了 0.95 | 降到 0.85；顺便关掉浏览器/微信等吃显存的程序 |
| `nvrtc: error: failed to open libnvrtc-builtins.so.13.0` | 该库在 venv 的 `site-packages/nvidia/cu13/lib/` 里，但 NVRTC 是 C 层加载器，不走 Python 路径 | `export LD_LIBRARY_PATH=$HOME/r3env/lib/python3.14/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH` |
| `driver.c: fatal error: Python.h: No such file or directory` | Triton 要 JIT 编译 C 扩展，缺 Python 开发头文件 | `sudo apt install -y python3-dev python3.14-dev` |
| `RuntimeError: NVRTCCompiler failed`（Blackwell sm_120 上） | CUDA graph / JIT warmup 期编译失败 | 加 `--enforce-eager` 关掉 CUDA graph |
| `routed_experts is None` | 模型不是 MoE / 开关没生效 | 确认模型名带 MoE，确认脚本里 `enable_return_routed_experts=True` |
| shape 对不上 | 末 token 无路由（行数少 1） | 这正是契约设计的一部分，不是 bug |
| 两次跑路由不一致 | 忘了 `VLLM_BATCH_INVARIANT=1` | 脚本开头已设，别删 |
| 下载的模型目录不对 | `--local-dir` 没指定 | 用 `--model` 指向你实际下载的目录 |

**挖坑心法**：正常跑通不是目的，**跑通过程中每一处「没想到会这样」的地方才是**。把你遇到的每个报错、每个意外行为，连同复现命令一起记下来。

---

## 第 7 步：OOM 预案（8GB 卡装 8GB 权重，几乎必撞）

> 前提：你现在下的是 **GPTQ-Int4** 版（Q4，约 8GB 权重）。8GB 卡 + 8GB 权重 + KV + 运行时开销 ≈ 9GB，**大概率 `CUDA out of memory`**。
> 这是 MoE，正好用「专家 offload」解：专家是稀疏激活的，把专家权重放 CPU 内存、命中才拉回 GPU，速度几乎不掉。

### 首选命令（专家 offload，直接跑）

```bash
# 先确认 host 内存够（offload 要占 CPU 内存，建议 ≥ 12GB）
free -h

python r3_verify.py \
  --model ~/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.95 \
  --cpu-offload-gb 6 \
  --cpu-offload-params experts \
  --offload-backend uva
```

**为什么这样写（三个踩空点，别省）：**

| 参数 | 作用 | 为什么不能省 |
|---|---|---|
| `--cpu-offload-params experts` | 只 offload 稀疏专家 | 不写它，vLLM 按层顺序 offload，会把每 token 都读的 attention/norm 也搬到 CPU，反而更慢还更 OOM |
| `--offload-backend uva` | 按需从 CPU 读，命中才拉 | 默认 auto 可能选 `prefetch`，prefetch 整层预取、把没被路由的专家也搬进来，省显存的效果全没了 |
| `--max-model-len 1024` | 砍 KV cache | KV 每 token 196KB，1024 上下文 ≈ 0.2GB，比 2048 省一半 |

### 验证 offload 真的生效了

启动日志里必须出现这一行，没有就是没 offload（`--cpu-offload-gb` 在旧版 vLLM 会被静默忽略，你的是 0.33.0，没问题）：

```
Total CPU offloaded parameters: xxx  (或 [PrefetchOffloader] Initialized)
```

### 兜底一：host 内存也被 OOM-killed

offload 的权重默认用 pinned memory（锁页内存），吃 host 内存。如果进程被系统 OOM-killer 杀掉：

```bash
export VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1
# 再跑上面的命令
```

### 兜底二：还是 OOM，就记成「第 5 个真实坑」

如果 `experts` offload 都救不回来，这本身就有价值——**「8GB 卡跑不动 14B MoE 的 GPTQ-Int4」就是一个真实坑**。记下：完整报错 + 复现命令 + `free -h` / `nvidia-smi` 输出，写进挖坑记录表。

---

## 附：如果 8GB 实在跑不动（Plan B）

不用灰心，这本身就是结论之一。两条退路：

1. **云 GPU**：租一块 A100（约 ¥10-20/小时），SSH 上去直接 `pip install vllm` 跑 BF16 原版，跳过量化这层复杂性，先把「标准环境下的坑」挖一遍。
2. **KTransformers**：CPU+GPU 混合跑 MoE（热专家放 GPU、冷专家放 CPU），专门为小显存跑大 MoE 设计，同样装在 WSL2 里。代价是引入「offload 带来的新坑」，但那些坑同样有价值。
