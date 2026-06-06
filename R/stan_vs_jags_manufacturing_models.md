# Stan vs JAGS — paired examples for manufacturing

This document contains two paired example models (Stan and JAGS) tailored to common manufacturing problems: **(A) Weibull reliability with partial pooling by production line** and **(B) Hierarchical p-chart with a change-point (drift)**. Each example includes: model code (Stan + JAGS), simulated-data generation, and short R glue to fit & check. Use these as drop-in templates for your shop-floor data.

------------------------------------------------------------------------

## A. Weibull time-to-failure with partial pooling (by line)

**Goal:** model time-to-failure `t_i` with Weibull shape `k` and scale `lambda[line[i]]`. Partial pooling across production lines (hierarchical) helps borrow strength for lines with few failures.

### Stan (continuous parameters — use NUTS)

``` stan
// weibull_hier.stan
data {
  int<lower=1> N;           // observations
  vector<lower=0>[N] t;     // failure times
  int<lower=1> L;           // number of lines
  int<lower=1,upper=L> line[N];
}
parameters {
  real<lower=0> k;                 // shape
  vector<lower=0>[L] lambda_raw;   // unscaled line-level scales
  real<lower=0> sigma_lambda;      // variability across lines
}
transformed parameters {
  vector<lower=0>[L] lambda;
  lambda = exp(log(mean(lambda_raw)) + sigma_lambda * (lambda_raw - mean(lambda_raw)));
}
model {
  // priors
  k ~ normal(0, 2) T[0, ];
  lambda_raw ~ normal(0, 1);
  sigma_lambda ~ normal(0, 1) T[0, ];

  // likelihood: Weibull parametrized with shape k and scale lambda[line]
  for (n in 1:N)
    t[n] ~ weibull(k, lambda[line[n]]);
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = weibull_lpdf(t[n] | k, lambda[line[n]]);
}
```

**Notes:** this parameterization keeps positivity and partial pooling. You can reparameterize (non-centered) if sampling issues arise.

### R (cmdstanr) skeleton

``` r
library(cmdstanr)
# simulate or load data: N, t, L, line
mod <- cmdstan_model('weibull_hier.stan')
fit <- mod$sample(data = data_list, chains=4, parallel_chains=4)
print(fit$summary())
# check divergences, Rhat, ESS; use posterior predictive checks
```

------------------------------------------------------------------------

### JAGS (supports similar hierarchical model)

``` jags
# weibull_hier.jags
model {
  for (n in 1:N) {
    t[n] ~ dweib(k, lambda[line[n]])
  }
  for (l in 1:L) {
    log(lambda[l]) <- mu + eta[l]
    eta[l] ~ dnorm(0, tau_eta)
  }
  mu ~ dnorm(0, 0.001)
  tau_eta <- pow(sigma_eta, -2)
  sigma_eta ~ dunif(0, 10)
  k ~ dgamma(0.1, 0.1)
}
```

**R (rjags) skeleton**

``` r
library(rjags)
jags.mod <- jags.model('weibull_hier.jags', data = data_list, n.chains=3)
update(jags.mod, 1000)
samples <- coda.samples(jags.mod, c('k','lambda','sigma_eta'), n.iter=5000)
plot(samples)
```

------------------------------------------------------------------------

## B. Hierarchical p-chart with a change-point (drift)

**Goal:** model defect counts `y[i]` out of `n[i]` inspections, with line-level random effects and a system-wide change in defect probability at some time. We'll provide two Stan strategies (smooth logistic transition — continuous change) and a JAGS strategy (discrete change-point).

### Stan: smooth transition (logistic) — avoids discrete latent

``` stan
// pchart_smooth.stan
data {
  int<lower=1> N;
  int<lower=0> y[N];
  int<lower=1> n_obs[N];
  int<lower=1> L;
  int<lower=1,upper=L> line[N];
  vector[N] time; // scaled time (e.g., 0..1)
}
parameters {
  real alpha0;                 // baseline log-odds
  vector[L] u;                 // line random effects (on log-odds)
  real<lower=0> sigma_u;
  real delta;                  // magnitude of change (log-odds)
  real<lower=0> kappa;         // steepness of logistic transition
  real midpoint;               // midpoint of transition in scaled time
}
transformed parameters {
  vector[N] logit_p;
  for (i in 1:N) {
    real s = inv_logit(kappa * (time[i] - midpoint));
    logit_p[i] = alpha0 + u[line[i]] + delta * s;
  }
}
model {
  // priors
  alpha0 ~ normal(0, 2);
  u ~ normal(0, sigma_u);
  sigma_u ~ normal(0, 1) T[0,];
  delta ~ normal(0, 1);
  kappa ~ normal(0, 5) T[0,];
  midpoint ~ beta(2,2); // assumes time scaled to (0,1)

  // likelihood
  for (i in 1:N)
    y[i] ~ binomial_logit(n_obs[i], logit_p[i]);
}
```

**R (cmdstanr)**: fit as before, inspect posterior of `midpoint`, `delta`; do posterior predictive checks across time slices.

------------------------------------------------------------------------

### JAGS: discrete change-point (explicit latent cp)

``` jags
# pchart_cp.jags
model {
  for (i in 1:N) {
    y[i] ~ dbin(p[i], n_obs[i])
    logit(p[i]) <- alpha0 + u[line[i]] + step_cp[i] * delta
    step_cp[i] <- equals(time_idx[i] > cp, 1)
  }
  for (l in 1:L) u[l] ~ dnorm(0, tau_u)
  tau_u <- pow(sigma_u, -2)
  sigma_u ~ dunif(0, 5)
  alpha0 ~ dnorm(0, 0.001)
  delta ~ dnorm(0, 0.001)
  cp ~ dunif(1, T_max)  # discrete uniform over possible time indices
}
```

**Notes:** JAGS' `equals()` and discrete `cp` make this straightforward. In practice, restrict `cp` domain (e.g., to plausible window) to improve mixing.

**R (rjags)**: supply `time_idx` as integer index; monitor `cp` and `delta`; mixing for `cp` may be slow — consider multiple chains and longer runs.

------------------------------------------------------------------------

## Practical tips & diagnostics

- **Stan:** watch for divergences, low E-BFMI, high R-hat; reparameterize (non-centered) if hierarchical scales mix poorly. Use posterior predictive checks and loo/waic for model comparison.
- **JAGS:** discrete cp and conjugate blocks are natural; watch autocorrelation, and use thinning/longer runs when needed. Use `coda` diagnostics.
- **Data prep:** scale time to (0,1) for logistic-midpoint priors; center covariates; check influence of prior choices in small-n regimes.

------------------------------------------------------------------------

## Next steps

- I can adapt either template to your real dataset (replace data generation with your `t`, `y`, `n`, `line`, `time` arrays), add simulated-data examples, or convert Stan code to brms-friendly formulas for faster prototyping.

*End of document.*
