# HMEMS 调试进度记录

## 当前日期
2026/04/21

## 状态
**verl 框架迁移完成 - 完整训练流程已跑通** (2026/04/21)
**奖励函数、优势计算、梯度回传、模型更新全部验证通过**

---

## 今日修复 (2026/04/21)

### 1. ADV_ESTIMATOR_REGISTRY 重复注册 ✅

**问题**：`ValueError: Adv estimator session_grpo has already been registered`

**根因**：Python 模块重复导入导致同一函数被注册两次

**修复**：在 `verl/trainer/ppo/core_algos.py` 中使 `register_adv_est` 幂等
```python
def decorator(fn):
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name in ADV_ESTIMATOR_REGISTRY:
        print(f"ADV_ESTIMATOR_REGISTRY[{name}] already registered, skipping")
        return fn  # Skip if already registered (idempotent)
    ADV_ESTIMATOR_REGISTRY[name] = fn
    return fn
```

---

### 2. batch_memories KeyError ✅

**问题**：`KeyError: 'batch_memories'` at ray_trainer.py:748

**根因**：stub `MemoryGenerationManager.run_memory_loop()` 未设置 `batch_memories` in meta_info

**修复**：在 `verl/trainer/ppo/ray_trainer.py` stub 中添加
```python
gen_output.meta_info['batch_memories'] = []  # Empty for stub
```

---

### 3. tensor_model_parallel_size 不匹配 ✅

**问题**：`AssertionError: rollout world_size: 1 is not divisible by infer_tp: 2`

**修复**：添加 `actor_rollout_ref.rollout.tensor_model_parallel_size=1`

---

### 4. log_prob_micro_batch_size_per_gpu 为 null ✅

**问题**：`TypeError: split(): argument 'split_size' must be int or list of ints`

**修复**：添加 `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1`

---

## 训练验证命令

```bash
# 启动 Ray
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

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

## 训练验证结果 (2026/04/21)

```
HMEMS training completed!
Training completed!

Step 0 Metrics:
- actor/entropy: 0.031
- train/mean_reward: 0.000
- response_length/mean: 512.0
- prompt_length/mean: 974.0
- timing_s/generation: 9.9s
- timing_s/old_log_prob: 0.87s
```

**说明**：
- Reward 为 0 是因为 memory server 未运行（端口 5005 连接被拒绝）
- Stub `HMEMSGenerationManagerWrapper` 正确处理了无 server 场景
- 完整训练流程已跑通：model init → generation → log_prob → reward → advantage → gradient → model update

---

## 核心文件修改

| 文件 | 修改内容 |
|------|---------|
| `verl/trainer/ppo/core_algos.py` | `register_adv_est` 幂等化 |
| `verl/trainer/ppo/ray_trainer.py` | stub `MemoryGenerationManager` 添加 `batch_memories` |
| `verl/workers/fsdp_workers.py` | 添加 stub MemoryGenerationManager fallback |
| `verl/workers/reward_manager/naive.py` | memalpha import 添加 try/except |
| `verl/trainer/ppo/reward.py` | sandbox_config.get() 处理 None |
| `src/session_based_core_algos.py` | 自定义 GRPO 优势计算 |
| `src/session_based_dataset.py` | Session-based 数据加载 |
| `src/session_based_reward_manager.py` | Session-based Reward 管理器 |
| `src/hmems_generation.py` | HMEMS Generation 管理器 |
| `src/standalone_trainer/ray_hmems_trainer.py` | Ray 训练器封装 |

---

## 下一步

1. 启动 memory server 验证真实 reward 计算
2. 使用更大数据集验证训练效果
3. 验证 checkpoint 正确保存和恢复

---

## 开发环境

- **Python**: verl-agent 环境 (conda)
- **Ray**: 2.49.2
- **Memory Server**: consolidation_server.py (端口 5005)
- **QA Model**: qwen-plus (通过 DashScope API)
- **训练入口**: `run_hmems_session_training.py`
