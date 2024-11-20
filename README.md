# Medical Evaluation Sphere

<img src="assets/medical_eval_sphere.png" alt="Medical Evaluation Sphere Logo" width="300"/>


## Medical QA Benchmark 🤗

```
from datasets import load_dataset

ds = load_dataset("lavita/medical-eval-sphere")

# loading the benchmark into a data frame
df = ds['medical_qa_benchmark_v1.0'].to_pandas()
```
