# Medical Evaluation Sphere

<img src="assets/medical_eval_sphere.png" alt="Medical Evaluation Sphere Logo" width="300"/>


## Medical QA Benchmark 🤗

```python
from datasets import load_dataset

ds = load_dataset("lavita/medical-eval-sphere")

# loading the benchmark into a data frame
df = ds['medical_qa_benchmark_v1.0'].to_pandas()
```

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
