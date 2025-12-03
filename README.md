# CKA-Agent: Bypassing LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search

<a href="https://arxiv.org/abs/2512.01353" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-CKA_Agent-red?logo=arxiv&style=for-the-badge" />
</a>
<a href="https://cka-agent.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Homepage-blue.svg?style=for-the-badge" />
</a>
<a href="https://github.com/Graph-COM/CKA-Agent" target="_blank">
    <img alt="GitHub code" src="https://img.shields.io/badge/💻_Code_GitHub-black.svg?style=for-the-badge" />
</a>
<a href="#cite" target="_blank">
    <img alt="Cite" src="https://img.shields.io/badge/📖_Cite!-lightgrey?style=for-the-badge" />
</a>
<a href="https://www.python.org/" target="_blank">
    <img alt="Python" src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge" />
</a>



## Overview
This repository contains the official implementation of **CKA-Agent**, a novel approach to bypassing the guardrails of commercial large language models (LLMs) through **harmless prompt weaving** and **adaptive tree search** techniques. 

![CKA-Agent](./assets/comparsion.png)


## Environment Setup
Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create env
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
uv pip install accelerate fastchat nltk pandas google-genai httpx[socks] anthropic
```

## Experiment Configuration

Configure your experiments by modifying the `config/config.yml` file. You can control the following aspects:

1.  **Test Dataset**: Choose from available datasets like `harmbench_cka` or `strongreject_cka`.
2.  **Target Models**: Select black-box or white-box models such as `gpt-oss-120b` or `gemini-2.5-xxx`.
3.  **Jailbreak Methods**: Enable and configure various implemented baseline methods.
4.  **Evaluations**: Define evaluation metrics and judge models like `gemini-2.5-flash`.
5.  **Defense Methods**: Apply different defense mechanisms as needed.

For detailed configuration instructions and examples, please refer to the [configuration README](config/README.md).

### Running Experiments

The `run_experiment.sh` script executes `main.py` to run the entire experiment pipeline (jailbreak and evaluation) by default.

```bash
./run_experiment.sh
```

You can modify the `run_experiment.sh` script or directly pass arguments to `main.py` to run specific phases:

-   `full`: Runs the entire pipeline (default).
-   `jailbreak`: Runs only the jailbreak methods.
-   `judge`: Runs only the evaluation on existing results.
-   `resume`: Resumes an interrupted experiment.

**Example (running only the jailbreak phase):**
```bash
python main.py --phase jailbreak
```


## Cite
If you find this repository useful for your research, please consider citing the following paper:

```bibtex
@misc{wei2025trojan,
      title={The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search}, 
      author={Rongzhe Wei and Peizhi Niu and Xinjie Shen and Tony Tu and Yifan Li and Ruihan Wu and Eli Chien and Pin-Yu Chen and Olgica Milenkovic and Pan Li},
      year={2025},
      eprint={2512.01353},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2512.01353}, 
}
```
