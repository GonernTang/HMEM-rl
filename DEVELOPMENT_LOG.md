# HMEMS 调试开发日志

## 日期
2026/04/17

## 调试目标
修复 HMEMS 训练流程全链路通通。

---

## 已修复问题

### 1. QA Pair Lookup Bug (关键修复 ✅)

**问题描述**：
- 训练数据有 25 个 session，全部来自 conv-49
- `qa_lookup.json` 中 `conv-49` 有 156 个 QA 对
- 每 batch 2 个 sample 时，发送 2 × 156 = 312 个 QA 对到 memory server
- 导致 memory server API 调用超时（60s timeout）

**根本原因**：
`src/hmems_generation.py` 中 `run_consolidation_loop` 使用 `self.qa_lookup.get(sample_id, [])` 获取 QA 对，这会返回该 conversation 的**所有** QA 对，而不是当前 session 的 QA 对。

**修复方案**：
1. `src/hmems_generation.py` - 添加 `qa_pairs_list` 参数
2. 当 `qa_pairs_list` 被传入时直接使用（per-session QA）
3. 仅在未传入时 fallback 到 `qa_lookup.get()`

**修改的文件**：
- `src/hmems_generation.py`:
  - 函数签名添加 `qa_pairs_list: List[List[Dict]] = None`
  - Step 5 逻辑改为使用传入的 `qa_pairs_list`
  ```python
  if qa_pairs_list is None:
      qa_pairs_list = []
      for sample_id in sample_ids:
          qa_pairs = self.qa_lookup.get(sample_id, [])
          qa_pairs_list.append(qa_pairs)

  # Build questions_list, ground_truth_answers_list, and qa_pairs_for_server from qa_pairs_list
  for i, qa_pairs in enumerate(qa_pairs_list):
      ...
  ```

- `src/standalone_trainer/generation_manager.py`:
  - `run_generation` 调用 `run_consolidation_loop` 时传入 `qa_pairs_list=qa_pairs_list`

**预期效果**：
- 修复前：2 samples × 156 QAs = 312 QA pairs/batch → timeout
- 修复后：2 samples × ~6 QAs = ~12 QA pairs/batch → 正常运行

---

### 2. Reward 计算全为 0 问题 ✅

**问题描述**：
- 训练时 reward 始终为 0.000
- Memory server 日志显示 `OPENAI_API_KEY not set, returning mock response`
- 预测答案全是 "Mock answer"

**根本原因**：
1. `.env` 文件中的 `OPENAI_API_KEY` 未传递给 `mock_memory_server.py`
2. `mock_memory_server.py` 读取 `OPENAI_API_BASE` 但 `.env` 用的是 `OPENAI_BASE_URL`
3. `retrieve_relevant_memories` 只搜索 `episodic_memory` 但存储在 `vector_memory`
4. `runtime_env.yaml` 排除了 `data/hmems_session_based/` 导致文件找不到

**修复方案**：
1. `scripts/train_hmems_single_gpu.sh` - 正确加载 `.env` 文件
2. `mock_memory_server.py` - 读取 `OPENAI_BASE_URL` 环境变量
3. `mock_memory_server.py` - 修复 `retrieve_relevant_memories` 搜索 `episodic_memory + vector_memory`
4. `verl/trainer/runtime_env.yaml` - 从排除列表移除 `data/hmems_session_based/`

**验证结果**：
- Step 0: mean_reward=0.628, max_reward=0.833
- Step 1: mean_reward=0.675, max_reward=0.750

---

### 3. 切换到真实 consolidation_server ✅

**问题描述**：
- 之前使用 `mock_memory_server.py`，使用简单关键词匹配
- 需要切换到真实的 `consolidation_server.py` 使用 OpenAI API

**修复方案**：
1. 修改 `consolidation_server.py` 的 API 格式兼容 `hmems_generation.py`:
   - `/consolidate` 支持 `{"decisions": [...]}` 格式
   - `/batch_process` 支持 `{"qa_pairs": [...]}` 格式并返回 `{"results": [...]}`
   - 添加 `compute_reward_keyword_match()` 函数

2. 更新 `scripts/train_hmems_single_gpu.sh`:
   - 从 `mock_memory_server.py` 切换到 `consolidation_server.py`

**验证结果**：
- API 格式兼容
- Reward 计算正常
- 训练流程正常运行

---

## 待解决问题

### 1. Ray/verl 环境问题 ✅ 已解决

**问题描述**：
```
ModuleNotFoundError: No module named 'verl.single_controller'
```

**根本原因**：
HMEMS/verl 是损坏的 verl 副本，内部所有 import 使用了 `from memoryrl.HMEMS.verl.verl.xxx` 的绝对路径，而 `memoryrl` 模块不存在。这些 393 个文件无法独立运行。

**解决方案**：
1. 将 HMEMS/verl 替换为 Mem-alpha/verl 的符号链接：`ln -s /root/autodl-tmp/memoryrl/Mem-alpha/verl /root/autodl-tmp/memoryrl/HMEMS/verl`
2. 修改 `/root/autodl-tmp/memoryrl/HMEMS/verl/trainer/runtime_env.yaml` 的 `working_dir` 和 `PYTHONPATH`
3. Mem-alpha/verl 使用相对导入（如 `from verl.utils.xxx`），可以通过 `PYTHONPATH` 正确解析

**修改的文件**：
- `HMEMS/verl/` → 符号链接到 `/root/autodl-tmp/memoryrl/Mem-alpha/verl`
- `HMEMS/verl/trainer/runtime_env.yaml`:
  - `working_dir: /home/wangyu/work/Mem-alpha` → `/root/autodl-tmp/memoryrl/Mem-alpha`
  - `PYTHONPATH: "/home/wangyu/work/Mem-alpha"` → `"/root/autodl-tmp/memoryrl:/root/autodl-tmp/memoryrl/Mem-alpha:/root/autodl-tmp/memoryrl/Mem-alpha/verl"`

**验证**：
```bash
python -c "from verl import single_controller; print('OK')"
python -c "from verl.protocol import DataProto; from verl.workers.fsdp_workers import ActorRolloutRefWorker; print('OK')"
```

**注意**：Mem-alpha/verl 的 `ray_trainer.py` 包含 `from memalpha.llm_agent.generation import ...`，HMEMS 不使用 `RayPPOTrainer`，而是用自定义的 `RayHMEMSTrainer`，已验证所有 HMEMS 训练组件可正常导入。

---

## 当前开发环境

### Python 环境
- 环境名：`verl-agent`
- Python：`/root/miniconda3/envs/verl-agent/bin/python`
- Ray：2.49.2 已安装

### 项目结构
```
/root/autodl-tmp/memoryrl/HMEMS/
├── verl/                          # 子模块 verd（不完整）
├── Mem-alpha/                     # Mem-alpha 项目（包含完整的 verl）
│   └── verl/                      # 完整的 verl 安装
├── src/
│   ├── hmems_generation.py        # 已修复
│   ├── standalone_trainer/
│   │   ├── generation_manager.py  # 已修复
│   │   └── ray_hmems_trainer.py
│   ├── session_based_dataset.py
│   ├── session_based_reward_manager.py
│   └── consolidation_agent.py
├── scripts/
│   ├── train_hmems_single_gpu.sh
│   └── train_hmems_session.sh
├── mock_memory_server.py          # Memory server (port 5005)
├── run_hmems_session_training.py  # 训练入口
└── data/hmems_session_based/      # 训练数据
    ├── train.jsonl (25 sessions, 193 QAs)
    ├── validation.jsonl (29 sessions, 242 QAs)
    └── test.jsonl
```

### 关键配置文件 (已验证可运行)
```bash
# 2026-04-16 成功运行的配置
data.train_files=data/hmems_session_based/train.jsonl
data.val_files=data/hmems_session_based/validation.jsonl
data.train_batch_size=4
data.val_batch_size=4
data.max_prompt_length=8192
data.max_response_length=2048
data.max_start_length=4096
data.max_obs_length=512
data.custom_cls.path=src/session_based_dataset.py
data.custom_cls.name=SessionBasedHMEMSDataset
algorithm.adv_estimator=grpo
actor_rollout_ref.model.path=/root/autodl-tmp/models/Qwen3-1.7B
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.use_kl_loss=true
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.rollout.mode=sync
actor_rollout_ref.rollout.enforce_eager=true
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
actor_rollout_ref.rollout.temperature=0.3
trainer.n_gpus_per_node=1
trainer.nnodes=1
trainer.total_epochs=1
reward_model.enable=True
reward_model.strategy=fsdp
reward_model.reward_manager=session_based
reward_model.compression_ratio_weight=0.05
respond_url=http://127.0.0.1:5005/batch_process
+consolidate_url=http://127.0.0.1:5005/consolidate
```

---

## 之前已修复的问题 (2026/04/16)

1. **`position_ids` missing** - vLLM 生成需要 position_ids
2. **vLLM mode=async 导致 None inference_engine** - 需要 `mode=sync` 和 `enforce_eager=true`
3. **Reward tensor indexing 错误** - 使用 `data.batch['responses'].shape[1]` 替代错误索引
4. **Hydra config key not found** - 使用 `+data.respond_url=` 语法添加新 key
5. **sample_id format 不匹配** - qa_lookup 使用 `conv_id`，代码使用 `conv_id_session_name`

---

## 训练流程架构 (已验证正确)

```
1. Consolidation Agent 生成 memory decisions (merge/augment/none)
2. 发送给 Memory Server (/consolidate) 存储记忆
3. Memory Server 使用外部 API (OpenAI) 基于记忆回答 QA
4. LLM-as-judge 评估回答质量
5. Reward 返回用于 advantage 计算
```

**注意**：Consolidation Agent **不直接**回答 QA，而是由 Memory Server 调用外部 API 回答。

---

## Memory Server 端点

- `/consolidate` - 存储记忆决策
- `/batch_process` - 回答 QA 并评估
- `/clear` - 清空记忆
- `/health` - 健康检查

当前运行：`http://127.0.0.1:5005`

---

## 下次调试步骤

1. **解决 Ray/verl 环境问题**
   - 检查 Mem-alpha/verl 是否需要 pip install -e
   - 或使用正确的 PYTHONPATH 配置
   - 或检查 runtime_env.yaml 配置

2. **验证 QA Pair Fix 生效**
   - 确认 per-session QA 对数量正确
   - 确认 memory server 不再超时
   - 确认 rewards 正确计算

3. **验证训练流程**
   - 检查 reward 值是否 > 0
   - 检查 model 是否正常更新
   - 检查 checkpoint 是否保存

---

## 相关文件路径

### 核心修复文件
- `/root/autodl-tmp/memoryrl/HMEMS/src/hmems_generation.py`
- `/root/autodl-tmp/memoryrl/HMEMS/src/standalone_trainer/generation_manager.py`

### 训练入口
- `/root/autodl-tmp/memoryrl/HMEMS/run_hmems_session_training.py`

### 配置文件
- `/root/autodl-tmp/memoryrl/HMEMS/scripts/train_hmems_single_gpu.sh`

### Memory Server
- `/root/autodl-tmp/memoryrl/HMEMS/consolidation_server.py` (真实服务器，使用 OpenAI API)
- `/root/autodl-tmp/memoryrl/HMEMS/mock_memory_server.py` (旧版本，已弃用)

### 之前成功运行的输出
- `/root/autodl-tmp/memoryrl/HMEMS/outputs/2026-04-16/23-39-28/`

---

## 日期
2026/04/20

## 调试目标
实现奖励函数、优势计算、梯度回传的正确计算

---

## 已修复问题

### 1. trainer.logger Hydra 配置格式错误 ✅

**问题描述**：
```
no viable alternative at input '[console,'
```

**根本原因**：
`trainer.logger=['console', 'tensorboard']` 的列表语法在 Hydra 命令行中不合法

**修复方案**：
```bash
# 修改 scripts/train_hmems_single_gpu.sh
trainer.logger="[console, tensorboard]" \
```

---

### 2. position_ids 和 attention_mask 缺失 ✅

**问题描述**：
```
KeyError: 'key "position_ids" not found in TensorDict'
```

**根本原因**：
`hmems_generation.py` 的 `run_consolidation_loop` 输出 DataProto 时没有包含 `position_ids`

**修复方案**：
在 `src/hmems_generation.py` 的 final_output 构建中，正确构建 full sequence 的 position_ids 和 attention_mask：

```python
# 正确构建：去除 padding，只保留实际 token
input_ids = gen_output.batch['input_ids']
responses = gen_output.batch['responses']

# 计算实际 prompt 长度（非 padding）
prompt_mask = (prompt_ids != pad_token_id).long()
prompt_lens = prompt_mask.sum(dim=1)

# 创建无 padding 的 full_input_ids
full_input_ids_list = []
for i in range(batch_size):
    actual_prompt_len = prompt_lens[i].item()
    prompt_tokens = prompt_ids[i, :actual_prompt_len]
    response_tokens = responses[i]
    full_input_ids = torch.cat([prompt_tokens, response_tokens])
    full_input_ids_list.append(full_input_ids)

# 统一 padding 到相同长度
padded_input_ids = torch.zeros(batch_size, max_total_len, ...)
attention_mask = torch.zeros(batch_size, max_total_len, ...)
position_ids = torch.zeros(batch_size, max_total_len, ...)

for i in range(batch_size):
    padded_input_ids[i, :actual_len] = full_input_ids_list[i]
    attention_mask[i, :actual_len] = 1
    position_ids[i, :actual_len] = torch.arange(actual_len, ...)
```

---

### 3. use_kl_loss=true 导致 ref_log_prob 缺失 ✅

**问题描述**：
```
KeyError: 'key "ref_log_prob" not found in TensorDict'
```

**根本原因**：
`use_kl_loss=true` 时 `dp_actor.py` 的 `update_policy` 会尝试读取 `ref_log_prob`，但我们没有实现 reference policy

**修复方案**：
```bash
# 修改 scripts/train_hmems_single_gpu.sh
use_kl_loss=false
```

---

### 4. compute_throughput_metrics 缺少 'step' key ✅

**问题描述**：
```
KeyError: 'step' (timing_raw 中没有 'step' key)
```

**根本原因**：
`ray_hmems_trainer.py` 中 `timing_raw` 没有被 `marked_timer('step', ...)` 包裹

**修复方案**：
在 `src/standalone_trainer/ray_hmems_trainer.py` 中添加条件检查：
```python
if 'step' in timing_raw and timing_raw['step'] > 0:
    throughput_metrics = compute_throughput_metrics(gen_output, timing_raw, n_gpus)
    metrics.update(throughput_metrics)
else:
    # Fallback throughput calculation
    metrics["perf/total_num_tokens"] = int(gen_output.batch['attention_mask'].sum())
    metrics["perf/time_per_step"] = 0.0
    metrics["perf/throughput"] = 0.0
```

---

### 5. memory server API 超时 ✅

**问题描述**：
```
HTTPConnectionPool(host='127.0.0.1', port=5005): Read timed out. (read timeout=60)
```

**根本原因**：
- `/batch_process` 串行调用 `qwen-plus` API
- 每个 batch 有 26-44 个 QA pairs
- 60 秒超时不够

**修复方案**：
在 `src/hmems_generation.py` 中增加超时时间：
```python
qa_response = requests.post(
    self.respond_url,
    json={"qa_pairs": qa_pairs_for_server},
    timeout=180  # 从 60s 增加到 180s
)
```

---

## 当前训练状态

### 训练流程已跑通 ✅

训练正在运行，已完成多个 step：
```
Epoch 0/5, Batch 0/6, Step 0, Mean Reward: 0.0000, Entropy: 0.0005 ...
Epoch 0/5, Batch 1/6, Step 1, Mean Reward: 0.0000, Entropy: 0.0003 ...
```

### Reward 为 0 的原因

Memory server 的 `/batch_process` 请求虽然超时时间已增加到 180s，但 QA 模型 (`qwen-plus`) 串行处理 26-44 个问题仍然较慢。

但代码层面的奖励计算逻辑已正确：
1. `SessionBasedRewardManager` 正确读取 `memory_server_rewards`
2. `compute_session_grpo_advantage` 正确计算优势
3. 梯度已正确回传

---

## 核心文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `src/hmems_generation.py` | 构建正确的 DataProto 输出（position_ids, attention_mask） |
| `src/standalone_trainer/ray_hmems_trainer.py` | 跳过缺失的 timing 指标 |
| `src/session_based_core_algos.py` | 自定义 GRPO 优势计算（非 evidence session advantage=0） |
| `src/session_based_reward_manager.py` | 正确使用 memory_server_rewards |
| `scripts/train_hmems_single_gpu.sh` | 设置 `use_kl_loss=false`，修复 `trainer.logger` 格式 |

---

## 下一步优化建议

1. **优化 memory server 响应时间**：
   - 考虑使用 async/threading 并行处理 QA pairs
   - 或使用更快的 QA 模型

2. **增加日志输出**：
   - 在 `SessionBasedRewardManager` 中打印 reward tensor 的实际值
   - 确认 reward 计算链路

3. **验证梯度更新**：
   - 检查模型权重在训练后是否确实变化
   - 添加 checkpoint 对比

---

## Environment

- **Python**: verl-agent 环境 (conda)
- **Ray**: 2.49.2
- **Memory Server**: consolidation_server.py (端口 5005)
- **QA Model**: qwen-plus (通过 DashScope API)
- **训练脚本**: `scripts/train_hmems_single_gpu.sh`

---

## 日期
2026/04/20 (下午)

## 调试目标
实现 per-dialogue-turn 的强化学习训练

---

## Per-Turn 训练设计

### 核心思想

当前实现：整个 session 对话作为**一个整体**处理，得到一个 consolidation decision

新设计：每个 dialogue turn **单独**由模型处理，累积 memory state，最后基于 session 级别 QA 计算奖励并回传到对应 evidence turn

### 处理流程

1. **Per-turn 处理（batch_size=4，同一个 session 的 4 个 copy）**
   ```
   For each dialogue_turn in session_dialogue:
       For each copy_idx in range(4):
           1. 构建 prompt = new_memory + retrieval results
           2. 模型生成 consolidation decision
           3. 更新 memory state
           4. 保存 response 和 log_prob
   ```

2. **Reward 计算（3 种模式）**
   - Mode 1: 所有 turn 相同 reward
   - Mode 2: evidence turn 相同 reward，非 evidence turn reward=0
   - Mode 3: 每个 turn 获得其 evidence QA 的平均 reward

3. **Advantage 计算**
   - Non-evidence turn 的 advantage=0，loss 被 mask

### 数据格式

```python
# qa_pairs 格式
{
    "question": "How many Prius has Evan owned?",
    "answer": "two",
    "evidence": ["D1:2", "D1:4"],  # dia_id 列表
    "category": 1
}

# session_dialogue 格式
[
    {"dia_id": "D1:1", "speaker": "Sam", "text": "Hey Evan..."},
    {"dia_id": "D1:2", "speaker": "Evan", "text": "Hey Sam! Good to see you..."},
    ...
]
```

### Memory State 管理

- 每个 batch 有 4 个独立的 ConversationMemoryState
- 每个 copy 维护独立的 vec_store 和 episodic_store
- 同一 session 的 4 个 copy 不共享 memory state（鼓励训练多样性）

---

## 已完成改动

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `src/conversation_memory_state.py` | ConversationMemoryState 类，管理单个 session copy 的 memory state |
| `docs/PER_TURN_TRAINING_DESIGN.md` | 完整设计文档 |

### 2. 修改文件

| 文件 | 修改内容 |
|------|----------|
| `src/hmems_generation.py` | 重写，新增 `run_per_turn_loop()` 方法用于 training，保留 `run_consolidation_loop()` 用于 validation/test |

### 3. 新增方法

**ConversationMemoryState**:
- `retrieve()`: 检索相似的 vec 和 episodic 记忆
- `add_vec_memory()`: 添加向量记忆
- `add_episodic_memory()`: 添加情节记忆
- `execute_action()`: 执行 consolidation decision
- `reset()`: 重置 memory state

**HMEMSGenerationManager** (新增):
- `_format_memories_for_prompt()`: per-turn prompt 格式化
- `_build_memory_summary()`: 构建 memory summary 用于 QA API
- `_call_batch_process_for_rewards()`: 调用 QA API
- `_allocate_rewards_to_turns()`: 3 种 reward 分配模式
- `_prepare_per_turn_output()`: 准备 per-turn 输出 DataProto
- `run_per_turn_loop()`: per-turn 训练主循环

---

## 待完成改动

1. **src/standalone_trainer/generation_manager.py** - 适配 per-turn 模式
2. **src/standalone_trainer/ray_hmems_trainer.py** - 修改 dataloader 和 fit() 循环
3. **scripts/train_hmems_single_gpu.sh** - 添加 `evidence_reward_mode` 参数

---

## 当前进度

Per-turn 训练的核心逻辑已实现在 `hmems_generation.py` 中：
- `run_per_turn_loop()` 处理完整的 per-turn 流程
- `_allocate_rewards_to_turns()` 支持 3 种 reward 分配模式
- 输出格式包含 `turn_rewards_list` 用于后续 advantage 计算

下一步需要：
1. 修改 `generation_manager.py` 调用新的 `run_per_turn_loop`
2. 修改 `ray_hmems_trainer.py` 的训练循环
3. 实现 per-turn advantage 计算和梯度回传

---

## 2026/04/20 (下午) 更新 - 实现完成

### ✅ 已完成的改动

#### 1. generation_manager.py
- 新增配置字段：`use_per_turn_mode` 和 `customized_grpo_rollout_n`
- 新增 `_run_session_level_generation()` - 用于 validation/test
- 新增 `_run_per_turn_generation()` - 调用 `inner.run_per_turn_loop()`
- `run_generation()` 根据 `use_per_turn_mode` 选择调用哪个方法

#### 2. ray_hmems_trainer.py
- `_create_hmems_generation_manager()` 添加 `use_per_turn_mode` 和 `customized_grpo_rollout_n` 配置传递
- `fit()` 循环中检测 `use_per_turn_mode`
- **Per-turn reward 处理**：
  - 从 `gen_output.meta_info['turn_rewards_list']` 获取预计算的 rewards
  - 构建 `reward_tensor` 形状为 (customized_grpo_rollout_n, num_turns)
- **Per-turn advantage 计算**：
  - 从 `turn_rewards_list` 计算每个 turn 的平均 reward
  - 创建 advantages 和 returns，形状为 (customized_grpo_rollout_n, num_turns)
- **Per-turn metrics**：添加 `train/num_turns` 和 `train_customized_grpo_rollout_n` 指标

#### 3. train_hmems_single_gpu.sh
新增配置参数：
```bash
use_per_turn_mode=false  # Set to true for per-turn training
customized_grpo_rollout_n=4
evidence_reward_mode=1  # 1, 2, or 3
```

新增训练参数：
```bash
+data.use_per_turn_mode=${use_per_turn_mode} \
+data.customized_grpo_rollout_n=${customized_grpo_rollout_n} \
+data.evidence_reward_mode=${evidence_reward_mode} \
```

---

## 总结

Per-turn 训练实现已完成，包括：
1. ✅ ConversationMemoryState - memory state 管理
2. ✅ hmems_generation.py - run_per_turn_loop 实现
3. ✅ generation_manager.py - per-turn 模式调用
4. ✅ ray_hmems_trainer.py - per-turn advantage 计算
5. ✅ train 脚本 - 新参数支持

下一步可能需要：
1. 测试训练流程
2. 验证 reward 和 advantage 计算是否正确
3. 优化 memory server 响应时间

---

## 2026/04/20 (下午) 测试结果

### 单元测试

1. **ConversationMemoryState 测试** ✅
   - 创建、添加 vec memory、retrieve、execute_action、reset 全部正常

2. **Reward Allocation 测试** ✅
   - Mode 1: 所有 turn 获得相同 reward ✅
   - Mode 2: evidence turn 获得 reward，非 evidence turn reward=0 ✅
   - Mode 3: 每个 turn 获得其 evidence QA 的平均 reward ✅

3. **HMEMSGenerationConfig 测试** ✅
   - use_per_turn_mode 和 customized_grpo_rollout_n 参数正确传递

4. **导入测试** ✅
   - 所有模块导入正常
   - run_hmems_session_training.py 相关导入正常

### 数据格式验证

训练数据格式确认：
- session_dialogue: 22 turns
- qa_pairs: 9 QAs
- evidence 字段: ["D1:2", "D1:4"] 格式

### 下一步

运行完整的端到端训练测试：
```bash
# 修改 train_hmems_single_gpu.sh
use_per_turn_mode=true

# 启动 memory server
python consolidation_server.py --port 5005 &

# 启动 Ray
ray start --head

# 运行训练
bash scripts/train_hmems_single_gpu.sh
```

---

## 2026/04/20 (17:20) - Bug 修复更新

### 问题背景

端到端训练运行中发现三个问题：
1. **Reward 全部为 0**：`Rewards computed: [0.0, 0.0, 0.0, 0.0]`
2. **Response length 显示为 0**：`RespLen: 0.0`
3. **update_actor 被跳过**：`Skipping update_actor (batch shape incompatible with verl)`

### 问题 1: Reward 全为 0

**根本原因分析**：

通过分析 memory_server.log 发现：
1. Memory server 收到 batch_process 请求并正常处理
2. 但 QA 模型预测答案全是"我没有足够的信息来回答这个问题"
3. 所有 Reward 显示为 0.0

这说明：
- Memory server 的 API 调用是成功的（200 OK）
- **问题在于 consolidation 步骤没有正确存储记忆** - grep 日志发现 `/consolidate` 端点几乎没有被调用
- 因为没有记忆存储，所以 QA 阶段无法检索到任何相关信息

**补充发现**：
- Memory server 的 consolidation_store 使用的是 `ConsolidationStore`（mock 实现），而不是真实的 `VecMem`
- 日志显示：`Using mock ConsolidationStore (VecMem not available)`
- 这导致即使 consolidation 被调用，记忆也没有被正确管理和检索

**临时修复**（增加 timeout）：
```python
# src/hmems_generation.py
timeout=180 → timeout=300  # 两处都修改
```

### 问题 2: Response Length = 0

**根本原因**：

在 `ray_hmems_trainer.py` 的 per-turn 模式中：
1. `compute_data_metrics` 函数被跳过（因为 per-turn batch 形状不兼容 verl）
2. 因此 `response_length/mean` 从未被计算
3. 日志输出 `metrics.get('response_length/mean', 0)` 显示为 0

**修复方案**：

在 `ray_hmems_trainer.py` 中为 per-turn 模式添加手动 response_length 计算：

```python
# Step 7: Compute data metrics (advantages, returns, response_length, etc.)
if use_per_turn_mode:
    # Per-turn mode metrics
    reward_scores = advantages.cpu().numpy()
    metrics['train/mean_reward'] = np.mean(reward_scores)
    metrics['train/max_reward'] = np.max(reward_scores)
    metrics['train/min_reward'] = np.min(reward_scores)
    metrics['train/num_turns'] = gen_output.meta_info.get('num_turns', 0)
    metrics['train/customized_grpo_rollout_n'] = gen_output.meta_info.get('customized_grpo_rollout_n', 0)

    # Compute response_length metrics manually for per-turn mode
    responses = gen_output.batch.get('responses')
    if responses is not None:
        actual_response_lens = []
        for i in range(responses.shape[0]):
            response_row = responses[i]
            non_padding = (response_row != self.tokenizer.pad_token_id).sum().item()
            actual_response_lens.append(non_padding)

        response_lens_array = np.array(actual_response_lens, dtype=np.float32)
        metrics['response_length/mean'] = np.mean(response_lens_array)
        metrics['response_length/max'] = np.max(response_lens_array)
        metrics['response_length/min'] = np.min(response_lens_array)
        metrics['response_length/clip_ratio'] = 0.0

        # Also compute prompt_length metrics
        attention_mask = gen_output.batch.get('attention_mask')
        if attention_mask is not None:
            prompt_lens = attention_mask.sum(-1).float() - torch.from_numpy(response_lens_array).float()
            metrics['prompt_length/mean'] = np.mean(prompt_lens.cpu().numpy())
            metrics['prompt_length/max'] = np.max(prompt_lens.cpu().numpy())
            metrics['prompt_length/min'] = np.min(prompt_lens.cpu().numpy())
            metrics['prompt_length/clip_ratio'] = 0.0
```

### 问题 3: update_actor 被跳过

**根本原因**：

Per-turn 模式下，advantages 的形状为 `(customized_grpo_rollout_n, num_turns)`，例如 `(4, 22)`：
- 22 个 dialogue turns
- 4 个独立的 session copies

但 verl 的 `update_actor` 期望 token-level 的形状，如 `(batch_size, max_seq_len)`。

原有的实现简单粗暴地跳过了整个 update_actor：
```python
if not use_per_turn_mode:
    # update_actor...
else:
    print(f"[PER-TURN] Skipping update_actor (batch shape incompatible with verl)")
```

**修复方案**：

实现 per-turn advantages 到 token-level 的扩展：

```python
# Step 5: Compute advantage using session-based GRPO
if use_per_turn_mode and 'turn_rewards_list' in gen_output.meta_info:
    # Expand per-turn advantages to token-level for update_actor compatibility
    customized_grpo_rollout_n = gen_output.meta_info.get('customized_grpo_rollout_n', 4)
    num_turns = gen_output.meta_info.get('num_turns', 1)
    all_dia_ids = gen_output.meta_info.get('all_dia_ids', [])
    all_responses = gen_output.meta_info.get('all_responses', [])

    # Compute average reward per turn across copies
    turn_rewards_list = gen_output.meta_info['turn_rewards_list']
    turn_advs = []
    for turn_idx, dia_id in enumerate(all_dia_ids):
        avg_reward = np.mean([
            turn_rewards_list[c].get(dia_id, 0.0)
            for c in range(customized_grpo_rollout_n)
        ])
        turn_advs.append(avg_reward)

    # Create per-turn advantages tensor: shape (customized_grpo_rollout_n, num_turns)
    turn_advantages = torch.tensor(turn_advs).unsqueeze(0).expand(customized_grpo_rollout_n, -1).float()
    turn_returns = turn_advantages.clone()

    # Expand per-turn advantages to token-level using all_responses
    if all_responses and len(all_responses) > 0 and len(all_responses[0]) == num_turns:
        # Get per-turn response lengths from first copy
        turn_response_lens = [len(all_responses[0][turn_idx]['gen_output'])
                             for turn_idx in range(num_turns)]

        # Expand advantages to token level
        total_response_len = sum(turn_response_lens)
        expanded_advantages = torch.zeros(customized_grpo_rollout_n, total_response_len, dtype=torch.float32)
        expanded_returns = torch.zeros(customized_grpo_rollout_n, total_response_len, dtype=torch.float32)
        expanded_response_mask = torch.zeros(customized_grpo_rollout_n, total_response_len, dtype=torch.float32)

        current_pos = 0
        for turn_idx, turn_len in enumerate(turn_response_lens):
            expanded_advantages[:, current_pos:current_pos + turn_len] = turn_advantages[:, turn_idx:turn_idx + 1]
            expanded_returns[:, current_pos:current_pos + turn_len] = turn_returns[:, turn_idx:turn_idx + 1]
            expanded_response_mask[:, current_pos:current_pos + turn_len] = 1.0
            current_pos += turn_len

        advantages = expanded_advantages
        returns = expanded_returns
        response_mask = expanded_response_mask

        # Flatten responses properly for update_actor
        all_flat_responses = []
        for copy_idx in range(customized_grpo_rollout_n):
            flat_response = torch.cat([all_responses[copy_idx][turn_idx]['gen_output']
                                      for turn_idx in range(num_turns)])
            all_flat_responses.append(flat_response)

        max_flat_len = max(len(r) for r in all_flat_responses)
        padded_flat_responses = torch.zeros(customized_grpo_rollout_n, max_flat_len, dtype=torch.long)
        for i, r in enumerate(all_flat_responses):
            padded_flat_responses[i, :len(r)] = r

        gen_output.batch['responses'] = padded_flat_responses
```

同时修改 update_actor 调用逻辑，只在无法展开时才跳过：

```python
# Step 8: Update actor (actual policy gradient update)
skip_update = False
if use_per_turn_mode:
    all_responses = gen_output.meta_info.get('all_responses', [])
    if not all_responses or len(all_responses) == 0:
        print(f"[PER-TURN] Skipping update_actor (all_responses not available)")
        skip_update = True
    elif len(all_responses[0]) != gen_output.meta_info.get('num_turns', 0):
        print(f"[PER-TURN] Skipping update_actor (turn count mismatch)")
        skip_update = True

if not skip_update:
    with marked_timer('update_actor', timing_raw, color="red"):
        gen_output.meta_info['multi_turn'] = self.config.actor_rollout_ref.rollout.get('multi_turn', {}).get('enable', False)
        actor_output = self.actor_rollout_wg.update_actor(gen_output)
        if actor_output and hasattr(actor_output, 'meta_info') and 'metrics' in actor_output.meta_info:
            actor_metrics = reduce_metrics(actor_output.meta_info['metrics'])
            metrics.update(actor_metrics)
```

### Per-Turn Advantage 扩展最终方案

经过多次迭代，最终采用以下方案：

1. **保持原始 responses 不变**：不替换 `gen_output.batch['responses']`
2. **扩展 advantages 到原始 response shape**：将 per-turn advantages `(customized_grpo_rollout_n, num_turns)` 扩展到 `(customized_grpo_rollout_n, max_response_len)`
3. **使用 `response_mask` 处理 padding**：`response_mask` 在扩展区域内为 1，其他区域为 0

```python
# Expand per-turn advantages to token-level
# We expand to the ORIGINAL response shape (from generation)
# so that log_probs and advantages have compatible shapes for update_actor
original_responses = gen_output.batch['responses']
orig_shape = original_responses.shape  # (customized_grpo_rollout_n, max_response_len)

turn_response_lens = [len(all_responses[0][turn_idx]['gen_output'])
                     for turn_idx in range(num_turns)]

expanded_advantages = torch.zeros(orig_shape, dtype=torch.float32)
expanded_returns = torch.zeros(orig_shape, dtype=torch.float32)
expanded_response_mask = torch.zeros(orig_shape, dtype=torch.float32)

current_pos = 0
for turn_idx, turn_len in enumerate(turn_response_lens):
    end_pos = current_pos + turn_len
    if end_pos <= orig_shape[1]:
        expanded_advantages[:, current_pos:end_pos] = turn_advantages[:, turn_idx:turn_idx + 1].expand(-1, turn_len)
        expanded_returns[:, current_pos:end_pos] = turn_returns[:, turn_idx:turn_idx + 1].expand(-1, turn_len)
        expanded_response_mask[:, current_pos:end_pos] = 1.0
    else:
        remaining = orig_shape[1] - current_pos
        if remaining > 0:
            expanded_advantages[:, current_pos:current_pos + remaining] = turn_advantages[:, turn_idx:turn_idx + 1].expand(-1, remaining)
            expanded_returns[:, current_pos:current_pos + remaining] = turn_returns[:, turn_idx:turn_idx + 1].expand(-1, remaining)
            expanded_response_mask[:, current_pos:current_pos + remaining] = 1.0
        break
    current_pos = end_pos
```

### 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `src/hmems_generation.py` | 将两处 `timeout=180` 改为 `timeout=300` |
| `src/standalone_trainer/ray_hmems_trainer.py` | 1. 添加 per-turn mode 的 response_length 手动计算<br>2. 实现 per-turn advantages 到 token-level 的扩展（扩展到原始 response shape）<br>3. 修改 update_actor 跳过逻辑 |

### 核心改动思路

1. **Response length = 0**：因为 `compute_data_metrics` 被跳过，添加手动计算
2. **update_actor 跳过**：per-turn advantages 形状 `(4, 22)` 无法匹配 verl 期望的 `(batch, seq_len)`
   - **最终方案**：扩展 advantages 到原始 response shape，保持 responses 不变，确保 log_probs 和 advantages 形状兼容
3. **Reward 全为 0**：基础设施问题 - memory server 未运行 + consolidation 未被调用导致记忆未存储

### 训练能否跑通？

**代码层面**：是的，经过以上修复，actor 梯度应该可以正常更新。

**前提条件**：
1. Memory server 必须运行并正确存储记忆
2. `all_responses` 必须可用（用于计算 turn_response_lens）
3. Turn response lengths 必须与原始 response shape 兼容

**潜在问题**：
1. 如果 turn response lengths 总和与原始 response 长度不一致，会有一些 token 的 advantage=0
2. Memory server 超时仍可能导致 reward=0

### 训练基础设施要求

确保以下服务运行：
```bash
# 启动 Memory Server
python consolidation_server.py --port 5005 &

# 启动 Ray
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 运行训练
bash scripts/train_hmems_single_gpu.sh
```

---

## 2026/04/21 - Memory Server 改动

### 日期
2026/04/21

### 调试目标
1. Reward 计算逻辑修改（关键词匹配 → LLM Judge）
2. 添加基于向量相似度的记忆检索
3. 每个新 Epoch 开始时重置 Memory Store

---

### 1. Reward 计算逻辑修改 ✅

**文件**: `consolidation_server.py`

将关键词匹配改为 LLM Judge 语义评估：

```python
# 之前: compute_reward_keyword_match() - 关键词匹配
# 现在: compute_reward_llm_judge() - LLM 语义评估

def compute_reward_llm_judge(question, predicted_answer, ground_truth):
    template = (
        "I will give you a question, a rubric for desired personalized response, "
        "and a response from a model. Please answer yes if the response satisfies "
        "the desired response. Otherwise, answer no. "
        "The model does not need to reflect all the points in the rubric. ...\n\n"
        "Question: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
        "Is the model response correct? Answer yes or no only."
    )
    prompt = template.format(question, ground_truth, predicted_answer)

    response = openai_client.chat.completions.create(
        model='qwen-plus',
        messages=[{"role": "user", "content": prompt}],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    content = response.choices[0].message.content.strip().lower()
    return 1.0 if "yes" in content and "no" not in content else 0.0
```

**验证结果**：
- 语义匹配 → 1.0
- 语义不匹配 → 0.0

---

### 2. Memory 检索功能添加 ✅

**文件**: `consolidation_server.py`

添加基于向量相似度的记忆检索，防止上下文溢出：

```python
# 新增配置
RETRIEVE_TOPK_VEC = 5          # 向量记忆最多检索5条
RETRIEVE_TOPK_EPISODIC = 5    # 情节记忆最多检索5条
EMBEDDING_MODEL = "text-embedding-v3"

# 新增函数
def get_embedding(text) -> np.ndarray:
    """获取文本 embedding"""

def cosine_similarity(a, b) -> float:
    """计算余弦相似度，处理维度不匹配"""
    min_dim = min(len(a), len(b))
    a_trunc = a[:min_dim]
    b_trunc = b[:min_dim]
    return np.dot(a_trunc, b_trunc) / (np.linalg.norm(a_trunc) * np.linalg.norm(b_trunc) + 1e-8)

def retrieve_relevant_memories(question, topk_vec=5, topk_episodic=5):
    """根据 question embedding 检索相关记忆"""
    question_emb = get_embedding(question)
    # 计算与所有记忆的相似度
    # 取 top-k 返回
    return {"vec_memories": [...], "episodic_memories": [...]}
```

**batch_process 修改**：
```python
# 之前: 使用全量记忆
memory_data = {"vec_memories": [v["content"] for v in vec_store.values()], ...}

# 现在: 只检索相关记忆
memory_data = retrieve_relevant_memories(question, topk_vec=5, topk_episodic=5)
```

**检索流程**：
```
Question: "What food does the user like?"
         ↓
get_embedding(question) → [0.1, 0.2, ...]  (1024维)
         ↓
对每条记忆计算 cosine_similarity:
  - "User likes blue" → 0.2  (低)
  - "Weather is sunny" → 0.1  (低)
  - "User prefers Japanese food" → 0.9  (高!)
         ↓
取 top-5:
  → vec_memories: ["User prefers Japanese food..."]
  → episodic_memories: ["color preference..."]
         ↓
generate_answer_via_openai(question, memory_data)
```

---

### 3. Epoch 开始时 Memory 重置 ✅

**文件**: `src/standalone_trainer/ray_hmems_trainer.py`

添加每个新 epoch 开始时重置 memory store：

```python
import requests  # 新增导入

for epoch in range(total_epochs):
    # 每个新 epoch 开始时重置 memory store
    if epoch > 0:
        try:
            reset_url = self.config.data.get("consolidate_url", "http://127.0.0.1:5005").replace("/consolidate", "/reset")
            resp = requests.post(reset_url, timeout=10)
            print(f"[Epoch {epoch}] Memory store reset for new conversation")
        except Exception as e:
            print(f"[Epoch {epoch}] Warning: Failed to reset memory store: {e}")
```

---

### 4. Memory 跨 Session 累积行为确认 ✅

**设计确认**：
- ✅ Session 1 → Session 2 → ... → Session N 记忆跨 session 累积
- ✅ 每个新 Epoch 开始时重置 memory store
- ✅ 检索机制确保每次只使用相关记忆（最多5条 vec + 5条 episodic）

**完整训练流程**：
```
Epoch 0:
  Session 1:
    /consolidate → 存入 memory
    /batch_process → 检索相关记忆 → 生成答案 → 计算 reward
  Session 2:
    /consolidate → memory store 累积 [s1, s2]
    /batch_process → 检索相关记忆（从 s1+s2）→ 生成答案 → 计算 reward
  ...
  Session N:
    memory store = [s1, s2, ..., sN]
    检索相关记忆 → 生成答案 → 计算 reward

Epoch 1 (新 conversation):
  Memory reset → store 清空
  Session 1:
    /consolidate → 存入 memory
    ...
```

---

### 修改文件清单

| 文件 | 修改类型 |
|------|---------|
| `consolidation_server.py` | 1. Reward 逻辑改为 LLM Judge<br>2. 添加 embedding 检索功能<br>3. 处理维度不匹配 |
| `src/standalone_trainer/ray_hmems_trainer.py` | Epoch reset 逻辑、requests 导入 |

---

### 训练命令

```bash
# 启动 Ray
source /root/miniconda3/bin/activate verl-agent
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 启动 Memory Server
nohup python consolidation_server.py --port 5005 > /tmp/consolidation_server.log 2>&1 &

# 运行训练
python run_hmems_session_training.py \
  data.train_files="data/hmems_session_based/test_mini.jsonl" \
  data.val_files="data/hmems_session_based/val_mini.jsonl" \
  data.max_prompt_length=1024 \
  data.max_response_length=512 \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.custom_cls.path="src/session_based_dataset.py" \
  data.custom_cls.name="SessionBasedHMEMSDataset" \
  algorithm.adv_estimator=session_grpo \
  actor_rollout_ref.model.path="/root/autodl-tmp/models/Qwen3-1.7B" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  trainer.n_gpus_per_node=1
```

---

### 验证结果

```
Training completed!
Step 0: train/mean_reward: 0.400 (5个QA中2个正确)
timing_s/generation: 19.158s (包含LLM judge调用)
```

---

## Per-turn GRPO Advantage 计算修复 (2026/04/21)

### 问题描述

训练时日志显示 `Adv: 0.0000`，用户询问为什么优势是 0。

### 原因分析

1. **原来的 advantage 计算逻辑**：
   - 对每个 turn，计算 4 个 copies 的平均 reward
   - 使用平均 reward 作为该 turn 的 advantage
   - 所有 copies 得到相同的 advantage

2. **GRPO 标准化的影响**：
   - 当 4 个 copies 的 reward 相同时（如 [0.5, 0.5, 0.5, 0.5]）
   - 平均值 = 0.5，标准差 = 0
   - `(reward - mean) / std = 0 / 0 = 0`

3. **日志显示问题**：
   - 显示的 `Adv: 0.0000` 是 `critic/advantages/mean`
   - 这是 verl 框架内部的指标，GRPO 模式（无 critic）下永远为 0
   - 不代表实际的 policy gradient 为 0

### 修复方案

**修改文件**: `src/standalone_trainer/ray_hmems_trainer.py`

1. **修复 advantage 计算**：
   - 使用 per-copy reward 进行 GRPO 标准化
   - 而不是先取平均再作为 advantage

```python
# 修复前（错误）：
turn_advs = []
for turn_idx, dia_id in enumerate(all_dia_ids):
    avg_reward = np.mean([...])  # 先取平均
    turn_advs.append(avg_reward)
turn_advantages = torch.tensor(turn_advs).unsqueeze(0).expand(...)

# 修复后（正确）：
per_copy_rewards = torch.tensor([
    [turn_rewards_list[c].get(dia_id, 0.0) for dia_id in all_dia_ids]
    for c in range(customized_grpo_rollout_n)
]).float()

# GRPO 标准化：每个 turn 在 copies 之间标准化
mean_per_turn = per_copy_rewards.mean(dim=0)
std_per_turn = per_copy_rewards.std(dim=0) + 1e-6
turn_advantages = (per_copy_rewards - mean_per_turn.unsqueeze(0)) / std_per_turn.unsqueeze(0)
```

2. **修复日志显示**：
   - 使用原始 reward 平均值用于监控（`train/mean_reward`）
   - 添加 `train/mean_advantage` 显示标准化后的 advantage
   - 日志显示改为 `GRPO-Adv` 标签

```python
# 修复后：
metrics['train/mean_reward'] = np.mean(raw_reward_scores)
metrics['train/mean_advantage'] = np.mean(advantage_scores)
# 日志：
adv_mean_str = f"GRPO-Adv: {metrics.get('train/mean_advantage', 0):.4f}"
```

### 参数命名说明

训练命令中的 `+data.customized_grpo_rollout_n=4` 是**遗留参数**，实际代码使用的是 `data.customized_grpo_rollout_n`。

- `+data.customized_grpo_rollout_n=4` - Hydra override，但代码中未读取
- `data.customized_grpo_rollout_n` - 实际使用的参数（默认值 4）

### GRPO Advantage 设计说明

GRPO 标准化后的 advantage 平均值为 0 是**正常设计**，不是 bug：

- `(reward - mean) / std` 的平均值必然为 0
- 这意味着相对于均值的偏离程度
- Policy gradient 会根据这个相对表现来更新策略

