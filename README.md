# Qwen3-VL-8B：MMK12 上的 vanilla ARS 与普通 ES

在自己的服务器上克隆后，进入仓库目录再执行后面的命令：

```bash
git clone https://github.com/XiangweiW/mmk12-zo.git
cd mmk12-zo
```

两份训练用同一个模型、同一个数据划分、同一种贪心解码和同一种 0/1 正确性奖励。差别只在参数更新。

- 模型：`Qwen/Qwen3-VL-8B-Instruct`
- 数据：`FanqingM/MMK12`。官方 train 15616 题，test 2000 题。
- 开发集：从 train 里用 seed 42 固定抽出 200 题，写入 `data/mmk12/splits.json`。
- 官方 test 只下载，不参与训练，也不参与选 checkpoint。

## 方法

Vanilla ARS

- 16 个各向同性随机方向，每个方向评估正负两侧，共 32 个候选。
- 所有候选用同一批题。
- 按 `max(R+, R-)` 选 top-8，下标小的优先。
- `sigma` 是这 8 个方向两侧平均奖励的总体标准差。
- `theta += lr / (8 * sigma) * sum (R+ - R-) * delta`
- `noise_std = 0.001`，`lr = 0.0001`。零方差或奖励差全为 0 时跳过更新。

普通 ES（ES at Scale 的做法）

- 30 个单侧扰动，没有正负成对。
- 奖励在这 30 个方向上做 z-score，全部参与更新，不取 top-k。
- 评估点是 `theta + sigma * epsilon`，`sigma = 0.001`。
- `theta += (alpha / 30) * sum Z_n * epsilon_n`，`alpha = 5e-4`。论文把 `1/sigma` 吸收进了 `alpha`。
- 种群标准差接近 0 时跳过更新。

两边的噪声都是 bf16，在 GPU 上按参数张量和 seed 重建，原地加减。更新先在 fp32 里累加，再写回 bf16。贪心解码：`temperature=0`，`top_p=1`，`top_k=-1`，`seed=42`。调度保持在一个进程里（`VLLM_ENABLE_V1_MULTIPROCESSING=0`），CUDA graph 开着。

奖励：最后一个 `\boxed{}` 和标准答案在归一化后一致，或两边都是数值且相对误差不超过 `1e-3`，记 1。没有框记 0。格式是否有框单独记录，不进更新。

## 环境

在驱动支持 CUDA 12.8 的机器上：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

8B 按每张卡一份模型来放。默认假设单卡约 80GB。显存不够时减小 `--batch-size`，或设置 `MMK12_MAX_MODEL_LEN`、`MMK12_GPU_MEMORY_UTILIZATION`。

## 数据

```bash
export MMK12_ROOT=$PWD
export HF_HOME=$PWD/cache/hf
python -m mmk12_zo.data
python -m mmk12_zo.checks
```

`python -m mmk12_zo.data` 会下载数据集，把图片存到 `data/mmk12/images/`，并写出划分。

## 训练

ARS 的 32 个候选要求 GPU 数量能整除 32，例如 8 或 4。ES 不要求整除。

```bash
export MMK12_ROOT=$PWD
export MMK12_MODEL=Qwen/Qwen3-VL-8B-Instruct
export HF_HOME=$PWD/cache/hf

python -m mmk12_zo.train --method ars --gpus 8 --steps 20 \
  --run-dir runs/ars

python -m mmk12_zo.train --method es --gpus 8 --steps 20 \
  --run-dir runs/es
```

默认每步 16 道题，`max_new_tokens=2048`。每 5 步以及最后一步在 200 题开发集上评测。日志在 `runs/<method>/steps.jsonl` 和 `dev_curve.jsonl`，bf16 checkpoint 在 `checkpoints/step_XXXX.safetensors`。中断后用同一个 `--run-dir` 再跑会从 `state.json` 继续。

每步生成量：ARS 是 `16 × 2 × batch_size`，ES 是 `30 × batch_size`。两边 batch 相同，但 ES 少做负侧，步数相同并不表示墙钟时间相同。
