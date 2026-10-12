# GeneaKV Phase 1

基于 Agent 会话谱系的选择性 CPU KV Cache 准入代码，面向 vLLM / vLLM-Ascend。

本仓库包含源码、测试和可复用脚本。模型、真实数据集、运行日志、实验结果和环境记录需要由使用者自行准备，不随代码分发。

## 代码结构

- `src/geneakv/model.py`：Agent 事件和 KV 策略的数据契约。
- `src/geneakv/lineage.py`：增量会话 DAG 和 HIGH / NORMAL / LOW 分类规则。
- `src/geneakv/connector.py`：复用原生 Ascend CPU Offload Worker，调整 Scheduler 侧保存准入。
- `scripts/`：合成事件生成、回放、服务启动、验证和分析工具。
- `tests/`：使用合成数据的功能测试。

## 本地功能测试

在仓库根目录执行：

```bash
PYTHONPATH=src:. python3 -m unittest discover -s tests -v
```

核心策略和回放测试不需要 NPU。原生 Connector 和设备验证脚本需要另行配置兼容的 vLLM、vLLM-Ascend、PyTorch-NPU 和 Ascend 运行环境。

## 合成事件回放

```bash
python3 scripts/generate_mini_traces.py --output datasets/synthetic/agent_events.jsonl
PYTHONPATH=src python3 scripts/replay_lineage.py \
  datasets/synthetic/agent_events.jsonl \
  --output results/lineage-policy-registry.jsonl
```

生成的数据和结果已加入忽略规则，不会随 Git 提交。

## 设备侧运行

Connector 面向 vLLM 0.21.0 / vLLM-Ascend 0.21.0rc1 的接口；其他版本需要检查原生 API 兼容性。

准备好模型和 Ascend 运行环境后设置 `MODEL_DIR`。启动脚本默认自动定位仓库根目录，也可用 `GENEAKV_ROOT` 覆盖。

```bash
# 将这个相对位置替换为自己准备的模型目录。
export MODEL_DIR="models/your-model"
bash scripts/run_native.sh
# 在另一个终端运行 CPU Offload 或 GeneaKV 服务：
bash scripts/run_offload.sh
# 或：
bash scripts/run_geneakv.sh
```

如果需要由 GeneaKV 启动脚本加载 Ascend 环境，可设置 `ASCEND_TOOLKIT_ENV`；也可在运行脚本前自行加载环境。通过 `DEVICE_ID`、`PORT`、`CPU_BYTES`、`MAX_MODEL_LEN` 等变量调整配置。默认服务仅绑定回环地址。

## 策略契约

请求通过 `kv_transfer_params.geneakv` 携带策略，例如：

```json
{
  "kv_transfer_params": {
    "geneakv": {
      "schema_version": 1,
      "session_id": "example-session",
      "node_id": "example-node",
      "event_seq": 1,
      "epoch": 0,
      "value_class": "HIGH",
      "protected_prefix_tokens": null,
      "reason_codes": ["TOOL_RESUME_EXPECTED"],
      "ttl_seconds": 300
    }
  }
}
```

HIGH 策略中的 `protected_prefix_tokens=null` 表示允许保存当前已确认的完整缓存块；显式数值用于限制保存边界。NORMAL、LOW 或无策略的请求不主动保存 CPU 副本。HBM 原生缓存和加载路径继续由原生实现管理。

`ttl_seconds`、`epoch` 和 `event_seq` 属于策略契约，当前 Connector 不据此主动删除旧 CPU 副本；CPU 容量仍由原生管理器控制。本阶段不包含主动 CPU→HBM Promotion。

## 自备 Trace 的回放与分析

回放工具使用 `kv-cache-tester` 风格的匿名块轨迹。数据来源需由使用者自行确认与准备；本仓库不附带任何真实请求或数据集快照。

`scripts/import_kv_cache_tester.py` 可转换 Trace，`scripts/build_interleaved_manifest.py` 可构建交错回放清单。使用各脚本的 `--help` 查看参数。运行矩阵前需自行设置 `TRACE_DIR`、`MANIFEST` 和模型配置。分析脚本只处理使用者提供的结果，不包含历史实验数值或性能结论。
