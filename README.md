# M PROMPTUNE
## Accelerates DSPy
Uses batch Expected Improvement to suggest multiple prompts to run in parallel. The mipro_optimizer_v2.py can parallelize but only at the the data evaluation level. The qEI sampler allows parallelization at the prompt level. Running prompts is definitely an IO bound process and you should definitely be running more threads than cores.
## Semantic Based
Tree Parzen Estimator [does't really handle](https://proceedings.mlr.press/v108/ma20a/ma20a.pdf) covariance between categorical variables. Instead of using Tree Parzen Estimator with a categorical distribution, qEI uses a Gaussian Process fit on **embeddings** of the possible combinations. This applies Bayesian Optimization at the **semantic** level in parallel.
## Forked Repository
Until my changes are merged with DSPy, I created a [forked repository](https://github.com/sign-of-fourier/dspy) that allows a sampler to be passed when initializing mipro_sampler_v2.
It is backwards compatible with DSPy and the same in every other way.
qEI.Sampler(max_space_size, n_batches, batch_size, min_cold_start)
- **max_space_size**: when sampling, the maximum number of points to consider.
- **n_batches**: when batching, the number of batches. *In this context, a batch is a group of suggestions.*
- **batch_size**: number in each batch
- **min_cold_start**: run with random selections until you get to this point

This example from DSPy optimizes instruction, few shot combinations
```
git git+https://github.com/sign-of-fourier/dspy.git      
```
Install my user defined sampler
```
pip install m-promptune
```
You will need tokens from [Open AI](https://platform.openai.com/api-keys) and [RapidAPI](https://rapidapi.com/info-FLGers_gH/api/batch-bayesian-optimization).
```
import os
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import MIPROv2

os.environ['OPEN_AI_KEY']='Your token here'
os.environ['X_RapidAPI_Key']='Your token here'
```
Configure the sampler to use qEI. The default is still the TPESampler.
```
sampler_config={'sampler': 'qei',
                'max_space_size': 100,
                'n_batches': 200,
                'batch_size': 4,
                'min_cold_start': 4}

```
Using example data (Grade school math 8K).
```
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
    n_jobs=2,
    **sampler_config
)
```
Specify the number of threads for parallelizing at the data evaluation level.
```
lm = dspy.LM('openai/gpt-4o-mini', api_key=os.environ['OPEN_AI_KEY'])
dspy.configure(lm=lm)
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="light",
    num_threads=2
)
```
