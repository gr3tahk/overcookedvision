# Overcooked VLM vs LLM Benchmark

Comparing VLM and LLM agents in the Overcooked cooperative cooking environment.

## Setup
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install overcooked-ai openai pillow numpy
```

## Running LLM Baseline
```bash
export OPENAI_API_KEY=your_key_here
python benchmark.py
```

## Running With A Local Model
```bash
pip install torch transformers accelerate sentencepiece
python benchmark.py --backend local
```

Optional: override the default local model.

```bash
python benchmark.py --backend local --local-model Qwen/Qwen2.5-3B-Instruct
```

## Collab-Overcooked Metrics

This repo implements the trajectory metrics from `colab.pdf`:

- `TES`: ordered trajectory efficiency against a reference action trajectory.
- `ITES`: incremental TES contribution of a new action or action batch.
- `PC`: per-agent progress completeness, averaged across agents.
- `IC` / `RC`: helper scoring for annotated collaboration initiations and responses once the LLM/VLM agents emit request/response events.

Run a local trajectory with metrics:

```bash
python benchmark.py --layout cramped_room --collect-trajectory --print-metrics
```

The replay JSON includes `symbolic_actions`, `agent_histories`, and `metrics`, so LLM-LLM and VLM-VLM runs can be compared using the same output schema.

## Project Structure
- `benchmark.py` — LLM agent and game loop
- `metrics.py` — TES, ITES, PC, IC, and RC metric helpers
- `vlm_agent.py` — VLM agent (Qwen-VL) [Josh]
- `test_llm.py` — quick test scripts
