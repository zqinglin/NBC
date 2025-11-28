# Neuro-Bayesian Continuum (NBC) Agent

一个基于冻结底座 + 多 LoRA 插槽 + 贝叶斯路由的持续学习智能体，实现交互期 TTT 觉醒学习与结束期睡眠巩固。

## 环境要求
- Python 3.10+（建议 3.10/3.11）
- Windows 或 Linux，建议有 NVIDIA GPU（支持 CUDA 11.8/12.1）
- 访问 Llama 3 需要 Hugging Face 账户与授权

## 安装步骤（Linux 服务器 / bash）
1. 检查 NVIDIA 驱动与 CUDA：
   ```bash
   nvidia-smi
   ```
   输出正常表示驱动就绪。CUDA 版本用于选择对应的 PyTorch 轮子（例如 12.1/11.8）。
2. 创建并激活虚拟环境（推荐 venv 或 conda）：
   - venv：
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - conda（可选）：
     ```bash
     conda create -n nbc python=3.10 -y
     conda activate nbc
     ```
3. 安装 PyTorch（按 CUDA 版本选择 Index URL）：
   - CUDA 12.1：
     ```bash
     pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
     ```
   - CUDA 11.8：
     ```bash
     pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
     ```
   - 无 GPU（CPU 版）：
     ```bash
     pip install torch torchvision torchaudio
     ```
4. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   ```
   Linux 下 `bitsandbytes` 将自动安装（已在 requirements.txt 做平台条件）。
5. 登录 Hugging Face（拉取 Llama 3）：
   ```bash
   huggingface-cli login
   ```

## 量化与显存建议
- 4-bit 量化依赖 `bitsandbytes`：Linux 支持较好；Windows 下若 4-bit 报错，可改用 16-bit。
- Llama-3-8B-Instruct：
  - 4-bit：显存约 8–10GB（视 KV 缓存与生成长度而定）
  - 16-bit：显存需 16GB+，建议使用更短 `max_new_tokens`

## 快速上手（Linux）
- 单次交互（默认尝试 4-bit）：
  ```bash
  python3 main.py --model meta-llama/Meta-Llama-3-8B-Instruct --quant 4bit --prompt "写一个二分查找函数"
  ```
- 若 4-bit 报错（驱动或 bnb 不兼容），改用 16-bit：
  ```bash
  python3 main.py --model meta-llama/Meta-Llama-3-8B-Instruct --quant 16bit --prompt "解释动态规划"
  ```
- 控制在线 TTT 强度与阈值：
  ```bash
  python3 main.py --ttt_steps 1 --threshold 1.5 --lr 5e-4 --max_new_tokens 128 --prompt "给出一个贪心算法示例"
  ```
- 执行睡眠巩固：
  ```bash
  python3 main.py --sleep --prompt "总结本轮学习"
  ```
- 后台运行（示例）：
  ```bash
  nohup python3 -u main.py --model meta-llama/Meta-Llama-3-8B-Instruct --quant 4bit --prompt "测试运行" > nbc.log 2>&1 &
  tail -f nbc.log
  ```

## 目录结构
- `agent.py`：容器与多插槽管理、TTT 觉醒与睡眠巩固
- `thalamus.py`：KAB-Thalamus 路由器（Beta 采样 + 语义锚点距离）
- `main.py`：命令行入口
- `requirements.txt`：依赖清单

## 常见问题（Linux）
- 4-bit 加载报错：
  - 检查 `nvidia-smi` 输出与 CUDA 版本；确认 `pip show bitsandbytes` 已安装。
  - 若仍不可用，切换 `--quant 16bit`，或升级驱动/使用 CUDA 兼容版本。
- 拉取 Llama 3 失败：
  - 确保已 `huggingface-cli login` 且账号获批 Llama 3 权限；或临时替换为兼容家族模型（如 Llama-3.1-Instruct）。
- 显存不足：
  - 降低 `max_new_tokens`、使用 4-bit、或选用更小的模型；同时减少 `do_sample` 参数或关闭采样以控制 KV 缓存占用。

## 提示
- 服务端推荐使用 `tmux`/`screen`/`nohup` 管理会话与日志。
- 根据任务不同，可调 `threshold`（惊讶门控）、`ttt_steps` 与 `decay`（睡眠动量），以平衡适应速度与稳定性。
