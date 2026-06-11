---

editor_options: 
  markdown: 
    wrap: 72
---

Given your domain — CNC milling process monitoring with an MLP estimating radial engagement (ae/D) from spindle current features — here is my expert recommendation on hyperparameter optimization strategies.
The short answer: use Optuna with Tree-structured Parzen Estimator (TPE) as the primary optimizer, combined with HalvingRandomSearchCV or Successive Halving for efficient early elimination. For your specific architecture (hidden layers, dropout, segment length), avoid grid search entirely and avoid Bayesian methods with Gaussian Process backends (too slow for the feature space you're working in).
Here is the reasoning, structured around your three hyperparameters:

## Why TPE (Optuna) is the right choice for your case
Hidden layers and units interact strongly with dropout. A 3-layer [512, 256, 128] architecture with 0.2 dropout behaves very differently from a 2-layer [256, 256] with 0.4. Grid search treats these as independent axes; TPE models their joint distribution and learns that, say, deeper networks need lower dropout to avoid collapsing the Kienzle-normalized features you're feeding it.

Segment length is your most physics-coupled hyperparameter. It determines how many tooth-passing cycles fit in a window, which directly affects the quality of your band energy and crest factor features. TPE will naturally discover the coupling between segment length and the frequency-domain features' SNR, whereas random search samples it obliviously.

Dropout specifically benefits from Bayesian optimization because it has a non-monotone relationship with validation loss: too low → overfitting to spindle noise artifacts, too high → the Monte Carlo uncertainty estimates become unreliable. TPE handles this kind of "sweet spot" search far better than grid or random.

The HyperbandPruner gives you successive halving "for free" inside Optuna — unpromising trials get pruned mid-training, so you effectively get both methods together. Set n_startup_trials=20 so TPE has enough initial random samples to build a reliable surrogate before it starts exploiting.

## One thing to watch for your domain
Because your features are Kienzle-normalized (torque residual, tooth-passing band energy, etc.), the optimal hidden layer widths will be much smaller than a naive image-classification MLP. Constrain units_l0 to the range [32, 256], not [32, 1024] — this tightens the search and prevents Optuna from wasting trials on over-parameterized configurations that will just memorize the training set's specific cutting conditions.

# Step 11
The notebook is built as Step 11, a direct continuation of your existing notebook. Every variable name, column name, and logic pattern matches what you already have. Here is what each section does and the key wiring decisions:

## Section 11.2
Section 11.2 — Segment rebuild helper. This was the most important thing to get right. Your **N_SEGMENTS = 4* is a fixed constant in Step 2, but the number of segments is also a hyperparameter to optimise. The helper re-runs your exact Step 2 loop — same *SinuTraceFile* loading, same *t_ramp_up* geometry, same friction offset via *mean_iq_frict*, same *scaler_X/scaler_y* — for any requested *n_segments* (2 to 8). That range is physically grounded: at **n=2** each segment captures half the cut (long, stable signal), at **n=8** you get 8× the samples but each segment is only ~1/8 of a cut (shorter, noisier).

## Section 11.3
Section 11.3 — Training helper. This is your Step 8 loop verbatim, with two added lines: *trial.report(val_loss, epoch)* and *trial.should_prune()*. These are the only changes — the HyperbandPruner in the study then kills bad trials at epochs 10, 30, and 90 before they reach epoch 150.
Section 11.4 — Flag SEARCH_N_SEGMENTS. Setting this to False skips the CSV re-scan and reuses your already-built matrices — useful for a quick first sweep to tune architecture alone. Set to True once you want the full joint search.
Section 11.7 — Retrain. The best_p.get("n_segments", N_SEGMENTS) call gracefully falls back to your original N_SEGMENTS = 4 if you ran with SEARCH_N_SEGMENTS = False, so the retrain cell works identically in both modes.
The recommended run order: first pass with SEARCH_N_SEGMENTS = False, n_trials=20, N_EPOCHS_TRIAL=60 as a quick sanity check. If that works, set SEARCH_N_SEGMENTS = True, n_trials=80, N_EPOCHS_TRIAL=150 for the real sweep.