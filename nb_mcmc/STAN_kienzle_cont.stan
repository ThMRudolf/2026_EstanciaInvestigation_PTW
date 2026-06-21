
data {
  int<lower=1> k;
  vector[k] Tc;
  vector[k] phi;
  real<lower=0> ap;
  real<lower=0> fz;
  int<lower=1> z;
  real<lower=0> rtool;
  real<lower=0> kappa;
  real m_kc;
  real<lower=0> sd_kc;
  real<lower=0> alpha_mc;
  real<lower=0> beta_mc;
  real phi_ent;
  real phi_exit;
}
parameters {
  real<lower=0,upper=1> mc;
  real<lower=0> kc11;
  real<lower=0> sigma;
}
model {
  mc    ~ beta(alpha_mc, beta_mc);
  kc11  ~ normal(m_kc, sd_kc);
  sigma ~ exponential(0.1);
  for (n in 1:k) {
    real Tc_pred = 0;
    for (i in 1:z) {
      real phi_eff = fmod(phi[n] + (i - 1) * 2.0 * pi() / z, 2.0 * pi());
      if (phi_eff >= phi_ent && phi_eff <= phi_exit)
        Tc_pred += kc11 * ap * pow(fz, 1 - mc) * pow(sin(kappa), mc)
                   * pow(sin(phi_eff), 1 - mc) * rtool;
    }
    Tc[n] ~ normal(Tc_pred, sigma);
  }
}
generated quantities {
  vector[k] log_lik;
  for (n in 1:k) {
    real Tc_pred = 0;
    for (i in 1:z) {
      real phi_eff = fmod(phi[n] + (i - 1) * 2.0 * pi() / z, 2.0 * pi());
      if (phi_eff >= phi_ent && phi_eff <= phi_exit)
        Tc_pred += kc11 * ap * pow(fz, 1 - mc) * pow(sin(kappa), mc)
                   * pow(sin(phi_eff), 1 - mc) * rtool;
    }
    log_lik[n] = normal_lpdf(Tc[n] | Tc_pred, sigma);
  }
}
