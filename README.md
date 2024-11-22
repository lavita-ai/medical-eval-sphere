# Medical Evaluation Sphere

<img src="assets/medical_eval_sphere.png" alt="Medical Evaluation Sphere Logo" width="300"/>


## Medical QA Benchmark 🤗

```python
from datasets import load_dataset

ds = load_dataset("lavita/medical-eval-sphere")

# loading the benchmark into a data frame
df = ds['medical_qa_benchmark_v1.0'].to_pandas()
```

## Notebooks Overview

The repository includes several Jupyter notebooks demonstrating various analyses and preprocessing steps. These notebooks are located in the [`notebooks`](https://github.com/lavita-ai/medical-eval-sphere/tree/main/notebooks) folder. Below is a brief description of each notebook:

- **[`data-preprocessing.ipynb`](https://github.com/lavita-ai/medical-eval-sphere/tree/main/notebooks/data-preprocessing.ipynb)**  
  Preprocessing queries, including deduplication, identifying medical questions, and filtering by language.

- **[`difficulty-level-analysis.ipynb`](https://github.com/lavita-ai/medical-eval-sphere/tree/main/notebooks/difficulty-level-analysis.ipynb)**  
  Analyzing the difficulty levels of medical questions.

- **[`llm-as-a-judge.ipynb`](https://github.com/lavita-ai/medical-eval-sphere/tree/main/notebooks/llm-as-a-judge.ipynb)**  
  Evaluating medical questions using an LLM-as-a-judge framework.

- **[`similarity-analysis.ipynb`](https://github.com/lavita-ai/medical-eval-sphere/tree/main/notebooks/similarity-analysis.ipynb)**  
  Performing inter- and intra-dataset similarity analysis and semantic deduplication.


#### Set up API Keys
Create a `.env` file in the root directory of the project and add the following lines:

```
ANTHROPIC_API_KEY=your_api_key_here
OPENAI_API_KEY=your_api_key_here
LABELBOX_API_KEY=your_api_key_here
```

### Citation
```bibtex
@article{hosseini2024benchmark,
  title={A Benchmark for Long-Form Medical Question Answering},
  author={Hosseini, Pedram and Sin, Jessica M and Ren, Bing and Thomas, Bryceton G and Nouri, Elnaz and Farahanchi, Ali and Hassanpour, Saeed},
  journal={arXiv preprint arXiv:2411.09834},
  year={2024}
}
```
