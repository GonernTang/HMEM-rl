# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HMEMS（Hierarchical Memory Enhancement System）是一个分层记忆增强系统，用于长期对话记忆管理，采用三层记忆架构：

1. **向量记忆（VecMem）**：原始对话片段的向量存储，使用余弦相似度检索
2. **情节记忆（Episodic Memory）**：通过 LLM 增强的高级记忆，自动合并相关原始记忆
3. **语义记忆（Semantic Memory）**：从情节记忆中提取的事实性知识

系统支持两种训练模式：
- **Session-Based GRPO Training**：基于强化学习的记忆 consolidation 策略优化
- **Standalone Training**：独立训练管道，支持多种训练框架

## 目录结构

```
HMEMS/
├── src/                          # 核心源代码
│   ├── vec_mem.py               # 向量存储实现
│   ├── episodic_memory.py       # 情节记忆
│   ├── semantic_memory.py       # 语义记忆
│   ├── pipeline.py              # 记忆处理流程
│   ├── run_experiments.py       # 实验入口（评估模式）
│   ├── consolidation_agent.py   # Consolidation Agent 实现
│   ├── reward_function.py       # Reward 计算函数
│   ├── hmems_dataset.py         # HMEMS 数据集
│   ├── hmems_generation.py      # Generation 管理器
│   ├── hmems_reward_manager.py  # HMEMS Reward 管理器
│   ├── session_based_*.py       # Session-based 训练组件
│   ├── standalone_trainer/      # Standalone 训练器
│   │   ├── ray_hmems_trainer.py # Ray 训练器
│   │   └── generation_manager.py
│   └── vector_store/            # 向量存储后端
│       ├── faiss_index.py       # FAISS 后端
│       └── flat_index.py        # 简单后端（调试用）
├── train_hmems_session.py       # Session-based 训练入口
├── run_hmems_session_training.py # Session-based Ray 训练入口
├── train_hmems_standalone.py     # Standalone 训练入口
├── consolidation_server.py       # Memory Server（端口 5005）
├── mock_memory_server.py         # Mock Memory Server（测试用）
├── scripts/                      # 训练脚本
└── data/                         # 数据目录
```

## 环境配置

### Conda 环境

```bash
conda create -n vecmem python=3.9 -y && conda activate vecmem
pip install -r requirements.txt
```

### .env 文件必需变量

```bash
OPENAI_API_KEY=<your_api_key>
OPENAI_BASE_URL=<api_base_url>      # Mock server 需要此变量
MODEL0=<embedding_model>             # 嵌入模型
MODEL1=<memory_ops_model>            # 记忆操作模型
MODEL2=<answer_gen_model>            # 回答生成模型
LOCOMO_PATH=<path_to_locomo10.json>
LOCOMO_EMBEDDING_PATH=<path_to_embeddings>
LOCOMO_INDEX_PATH=<path_to_indices>
LOCOMO_RES_PATH=<path_to_answers>
LOCOMO_SCORE_PATH=<path_to_scores>
# 可选：第三方 API 端点
M0_BASE_URL=<embedding_endpoint>
M1_BASE_URL=<memory_ops_endpoint>
M2_BASE_URL=<answer_gen_endpoint>
```

## 训练模式

### 1. Session-Based GRPO Training（强化学习）

使用 verl (from Mem-alpha) 进行强化学习训练，训练记忆 consolidation 策略。

**核心组件**：
- `SessionBasedHMEMSDataset`：加载 session 级数据，每个 session 包含对话和 QA pairs
- `SessionBasedRewardManager`：计算 QA 准确率 + 压缩率 reward
- `compute_session_grpo_advantage`：自定义 GRPO，非 evidence session 的 advantage = 0

**启动训练**：
```bash
# 启动 Memory Server
python consolidation_server.py --port 5005 &

# 启动 Ray
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 运行训练
python run_hmems_session_training.py \
    --model_path /path/to/model \
    --train_data data/hmems_session_based/train.jsonl \
    --val_data data/hmems_session_based/validation.jsonl
```

**关键文件**：
- `src/session_based_dataset.py` - Session-based 数据集
- `src/session_based_core_algos.py` - 自定义 GRPO 算法
- `src/session_based_reward_manager.py` - Session-based Reward 管理器
- `src/hmems_generation.py` - HMEMS Generation 管理器

### 2. Standalone Training

独立训练管道，使用 Ray 作为计算后端。

**启动训练**：
```bash
python train_hmems_standalone.py --config config.yaml
```

**关键文件**：
- `src/standalone_trainer/ray_hmems_trainer.py` - Ray 训练器
- `src/standalone_trainer/generation_manager.py` - Generation 管理器

### 3. 评估模式（无需训练）

```bash
# 初始化环境
python run_experiments.py --init_env

# 运行实验
python run_experiments.py \
    --min_aug_count 3 \
    --min_relevant_score 0.7 \
    --retrieve_raw_topk 5 \
    --retrieve_aug_topk 5 \
    --output_file results.json

# 仅评估已有结果
python run_experiments.py --eval_only --output_file <path_to_results.json>
```

## 核心架构

### 三层记忆系统数据流

**记忆添加流程**：
1. 新记忆先与情节记忆比较相似度（`try_merge_new_memory`）
2. 如不需合并，从向量存储检索相关记忆
3. 相关记忆数 >= `min_aug_count` 时触发 LLM 生成情节记忆
4. 被增强的原始记忆从向量存储移除
5. 可选：从情节记忆提取语义记忆

**记忆检索流程**：
1. 并行从向量存储和情节记忆检索
2. 可选：从语义记忆检索相关事实
3. 可选：迭代检索优化查询
4. 使用检索到的上下文生成答案

### VecMemConfig 关键参数

```python
@dataclass
class VecMemConfig:
    min_aug_count: int = 3           # 触发记忆增强的最小相关记忆数
    min_relevant_score: float = 0.7   # 记忆相关性阈值
    merge_with_aug_thresh: float = 0.85  # 与情节记忆合并的阈值
    retrieve_raw_topk: int = 5        # 从向量存储检索的数量
    retrieve_aug_topk: int = 5        # 从情节记忆检索的数量
    enable_iter_anwser: bool = False  # 迭代回答
    enable_semantic_memory: bool = False  # 语义记忆
```

### 提示词系统

所有提示词在 `src/prompt.py` 中定义，使用 Jinja2 模板：
- `MEMORY_AUGMENT_PROMPT`：生成情节记忆
- `MEMORY_AUGMENT_MERGE_PROMPT`：合并情节记忆
- `ANSWER_PROMPT_VECMEM`：使用向量和情节记忆生成答案
- `ITERATIVE_ANWSER_PROMPT_VECMEM`：迭代检索
- `SEMANTIC_EXTRACTION_PROMPT`：从情节记忆提取语义
- `ANSWER_PROMPT_WITH_SEMANTIC`：包含语义记忆的答案生成

### 向量存储

开发/测试用 `FlatIndex`（简单可调试），生产环境用 `FAISSIndex`（更快检索）。切换修改 `src/vec_mem.py` 和 `src/pipeline.py` 中的导入。

## Consolidation Agent

Agent 使用特殊 token 进行函数调用：`FN_NAME:` / `FN_ARGS:` / `FN_RESULT:` / `FN_EXIT`

三种 consolidation 操作：
- `merge`：合并相关记忆
- `augment`：生成新的情节记忆
- `none`：不进行操作

## 评估指标

三种评估指标：BLEU-1、F1 Score、LLM Judge（0-1 分数）

按五类分别统计：
- Category 1：多跳推理（multihop）
- Category 2：时序推理（temporal）
- Category 3：开放域（open_domain）
- Category 4：单跳推理（singlehop）
- Category 5：对抗性问题（adversarial，默认跳过）

## 开发注意事项

### 调试单个对话

```bash
python run_experiments.py --conv_limit 1 ...
```

### 检查记忆内容

在 `run_experiments.py` 的 `FINAL_RESULTS` 中保存了向量存储、情节记忆、语义记忆状态及每个问题的答案。

### API 调用失败

检查 `.env` 中的 `M1_BASE_URL` 和 `M2_BASE_URL` 配置是否正确。

### verl 依赖

HMEMS 使用 Mem-alpha 的 verl（通过符号链接）：
```
HMEMS/verl -> Mem-alpha/verl
```

如果 verl 导入有问题，检查符号链接是否正确。

### Session-Based Training 数据格式

```json
{
  "conv_id": "xxx",
  "session_name": "session_1",
  "session_dialogue": [...],
  "qa_pairs": [
    {"question": "...", "answer": "..."}
  ],
  "is_evidence_session": true
}
```

### Memory Server 端点

- `POST /consolidate`：执行 consolidation 决策
- `POST /batch_process`：批量处理 QA
- `POST /execute_tool`：执行工具调用

## Token 使用统计

```bash
python run_experiments.py --enable_stat ...  # 统计结果保存为 _token_stats.json
```
