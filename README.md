# M PROMPTUNE
## Accelerates DSPy
Uses batch Expected Improvement to suggest multiple prompts to run in parallel. The mipro_optimizer_v2.py can parallelize but only at the the data evaluation level. The qEI sampler allows parallelization at the prompt level. Running prompts is definitely an IO bound process and you should definitely be running more threads than cores.

## Semantic Based

Instead of using Tree Parzen with a categorical distribution, qEI uses a Gaussian Process fit on **embeddings** of the possible combinations.


## Forked Repository
Until my changes are merged with DSPy, I created a [forked repository](https://github.com/sign-of-fourier/dspy) that allows a sampler to be passed when initializing mipro_sampler_v2.
Use this one instead. It is backwards compatible with DSPy and the same in every other way.
```
from mpromptune import qEI
import os
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import MIPROv2
import random
import optuna
import numpy as np
```
You will need tokens from Open AI and [RapidAPI](https://rapidapi.com/info-FLGers_gH/api/batch-bayesian-optimization).
Pass the sampler wheninitialized and specify the number of prompts to run in parallel.
```
gsm8k = GSM8K()
sampler= qEI.Sampler(40, 400, 4)
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
    sampler=sampler,
    n_jobs=2
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
