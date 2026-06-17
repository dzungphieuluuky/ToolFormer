# ToolFormer
**Telecom Tool-Calling with Reinforcement Learning**  
Fine-tuning Qwen3-4B for accurate function calling in network operations,
comparing RC-GRPO, AWPO, and GVPO algorithms built on Unsloth's GRPO implementation.

## Quick Start

```bash
# 1. Create environment
conda create -n unsloth_telco python=3.12
conda activate unsloth_telco
pip install -r requirements.txt

# 2. Set API key (for dataset generation)
export OPENAI_API_KEY="sk-..."        # or ANTHROPIC_API_KEY / TOGETHER_API_KEY

# 3. Place your function schema
cp /path/to/function_schema.json data/processed/function_schema.json

# 4. Prepare data (generates ~2400 synthetic samples)
python scripts/prepare_data.py --schema data/processed/function_schema.json

# 5. SFT warm-start (recommended)
python scripts/run_sft.py

# 6. RL training
python scripts/run_rc_grpo.py
python scripts/run_awpo.py
python scripts/run_gvpo.py

# 7. Evaluate all models
python scripts/run_evaluation.py \
    --models outputs/rc_grpo_model outputs/awpo_model outputs/gvpo_model

# 8. Ablation studies
python scripts/run_ablation.py