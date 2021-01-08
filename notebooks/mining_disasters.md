# Coal Mining Disasters in the UK

Consider the following time series of recorded coal mining disasters in the UK from 1851 to 1962 (Jarrett, 1979). The number of disasters is thought to have been affected by changes in safety regulations during this period.

Next we will build a model for this series and attempt to estimate changes in the underlying risk of disasters.

```python
import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
```

```python
import warnings
from datetime import datetime

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import pymc3 as pm
import seaborn as sns

cov = pm.gp.cov

sns.set_context("notebook")

warnings.simplefilter("ignore")

%matplotlib inline
```

```python
# Time series of recorded coal mining disasters in the UK from 1851 to 1962
disasters_data = pd.read_csv("../data/mining-disasters.csv", header=None)[0].values
year = np.arange(1851, 1962)
year_ind = (year - year.min()).reshape(-1, 1)
```

```python
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(year, disasters_data, color="#348ABD", alpha=0.65, width=0.7)
ax.set_xlim(year[0], year[-1] + 1)
ax.set_ylabel("Disaster count")
ax.set_xlabel("Year");
```

![png](mining_disasters_files/mining_disasters_4_0.png)

```python
with pm.Model() as disasters_model:

    ρ = pm.Exponential("ρ", 1)
    η = pm.Exponential("η", 1)

    K = η ** 2 * cov.ExpQuad(1, ρ)
    gp = pm.gp.Latent(cov_func=K)
```

```python
with disasters_model:

    f = gp.prior("f", X=year_ind)
```

```python
with disasters_model:

    λ = pm.Deterministic("λ", pm.math.exp(f))

    confirmation = pm.Poisson("confirmation", λ, observed=disasters_data)
```

```python
with disasters_model:

    trace = pm.sample(1000, tune=1000, chains=2)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [f_rotated_, η, ρ]

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 07:04<00:00 Sampling 2 chains, 5 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 435 seconds.
    There were 5 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 25% for some parameters.

```python
az.plot_posterior(trace, var_names=["ρ", "η"]);
```

![png](mining_disasters_files/mining_disasters_9_0.png)

```python
years = year - year.min()
```

```python
with disasters_model:
    y_pred = pm.sample_posterior_predictive(trace, vars=[f], samples=1000)
```

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='1000' class='' max='1000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1000/1000 00:04<00:00]
</div>

```python
fig, ax = plt.subplots(figsize=(12, 5))

pm.gp.util.plot_gp_dist(ax, np.exp(y_pred["f"]), year[:, None])
sns.regplot(year, disasters_data, fit_reg=False, ax=ax)
ax.set_xlim(year.min(), year.max())
ax.set_ylabel("Disaster rate");
```

![png](mining_disasters_files/mining_disasters_12_0.png)

```python

```
