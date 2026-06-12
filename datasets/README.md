---
license: apache-2.0
task_categories:
  - question-answering
language:
  - en
tags:
  - rag
  - retrieval-augmented-generation
  - multi-hop-qa
  - agentic-rag
  - benchmark
pretty_name: "A-RAG Benchmark Datasets"
size_categories:
  - 1K<n<10K
---

# A-RAG Benchmark Datasets

Unified benchmark datasets for evaluating [A-RAG](https://github.com/Ayanami0730/arag) (Agentic Retrieval-Augmented Generation).

📄 **Paper**: [A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces](https://arxiv.org/abs/2602.03442)

## Dataset Description

This repository contains five multi-hop QA benchmark datasets, each with a document corpus (`chunks.json`) and evaluation questions (`questions.json`). These datasets are reformatted into a unified format for A-RAG evaluation.

### Included Datasets

| Dataset | Questions | Chunks | Description |
|---------|-----------|--------|-------------|
| `musique` | 1,000 | 1,354 | Multi-hop QA (2-4 hops) |
| `hotpotqa` | 1,000 | 1,311 | Multi-hop QA |
| `2wikimultihop` | 1,000 | 658 | Multi-hop QA |
| `medical` | 2,062 | 225 | Domain-specific (medical) QA |
| `novel` | 2,010 | 1,117 | Long-context (literary) QA |

### Data Sources

These datasets are **not** originally created by us. We unified them into a consistent format for A-RAG evaluation:

- **MuSiQue, HotpotQA, 2WikiMultiHopQA**: Reformatted from [Zly0523/linear-rag](https://huggingface.co/datasets/Zly0523/linear-rag), which follows the LinearRAG experimental setup.
- **Medical, Novel**: Reformatted from [GraphRAG-Bench](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench).

Please cite the original dataset papers if you use them in your research (see below).

## File Format

### chunks.json

```json
[
  "0:chunk text content here...",
  "1:another chunk text content...",
  ...
]
```

Each entry is a string in `"id:text"` format, where `id` is the chunk index.

### questions.json

```json
[
  {
    "id": "musique_2hop__13548_13529",
    "source": "musique",
    "question": "When was the person who ...",
    "answer": "June 1982",
    "question_type": "",
    "evidence": ""
  },
  ...
]
```

## Quick Start with A-RAG

```bash
# Clone A-RAG
git clone https://github.com/Ayanami0730/arag.git && cd arag
uv sync --extra full

# Download dataset
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Ayanami0730/rag_test', repo_type='dataset', local_dir='data')
"

# Build index & run
uv run python scripts/build_index.py --chunks data/musique/chunks.json --output data/musique/index --model sentence-transformers/all-MiniLM-L6-v2
```

See the [A-RAG repository](https://github.com/Ayanami0730/arag) for full instructions.

## Citation

If you use these datasets with A-RAG, please cite:

```bibtex
@misc{du2026aragscalingagenticretrievalaugmented,
      title={A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces},
      author={Mingxuan Du and Benfeng Xu and Chiwei Zhu and Shaohan Wang and Pengyu Wang and Xiaorui Wang and Zhendong Mao},
      year={2026},
      eprint={2602.03442},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2602.03442},
}
```

Please also cite the original dataset sources:

```bibtex
@article{trivedi2022musique,
  title={MuSiQue: Multihop Questions via Single Hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  year={2022}
}

@article{yang2018hotpotqa,
  title={HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W and Salakhutdinov, Ruslan and Manning, Christopher D},
  year={2018}
}

@article{ho2020constructing,
  title={Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps},
  author={Ho, Xanh and Nguyen, Anh-Khoa Duong and Sugawara, Saku and Aizawa, Akiko},
  year={2020}
}

@article{xiang2025graphragbench,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  year={2025}
}
```
