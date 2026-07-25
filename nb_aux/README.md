# nb_aux

Auxiliary / exploratory notebooks: data-format conversion (Hermle JSON exports →
CSV/Parquet), one-off segmentation and event-extraction work on specific recordings,
and a couple of unrelated paper-figure notebooks. Not part of the main modeling
pipelines (`nb_ae/`, `nb_mcmc/`, `Kienzle_model_simulation_pipeline/`).

## Notebooks

### `json2csv.ipynb`
Converts the MongoDB-exported `*.processed.json` files in `data/Hermle/20260623/`
(the `Kraftmessung`/`Vary` force-measurement tests) into per-record CSVs under
`data/converted/20260623/`. This is a different, older schema than the axon-logger
notebooks below — one JSON file holds many fixed-length-array records (200 or 1000
samples), flattened here into long format with a reconstructed per-sample time axis
(`add_time_calc`, since the raw `time` field only changes once per record).

### `axon_logger_json2df.ipynb`
Converts a Hermle axon-logger `*.raw.json` export (interactively picked via
`FileChooser`, default `data/Hermle/20260630/axon_logger_ptw_Lev006.raw.json`) into one
Parquet file per signal channel (`force_0/1/2`, `acceleration_0/1/2`, `distance_0/1/2`,
`machine_0`). Loads the whole file with `json.load` (fine up to ~200 MB — no
streaming). Also downsamples the force channels to 1 kHz for plotting. This is the
notebook that defines the core parsing helpers (`extract_block_meta`,
`build_waveform_df`, `_unwrap_number`, `flatten_machine_sample`, `build_machine_df`)
reused by the two lev008 notebooks below.

### `axon_logger_json2df_.ipynb` — redundant, see below
Same notebook as `axon_logger_json2df.ipynb` (identical markdown and code, cell for
cell), just pointed at a different default file
(`data/Hermle/20260629/axon_logger_ptw_00test03.raw.json`) and missing the last two
cells (the downsample/plot section). Looks like an earlier save of the same working
file rather than a distinct notebook.

### `axon_logger_lev008_process_cut_segmentation.ipynb`
Full pipeline for the 2 GB `axon_logger_ptw_lev008.raw.json` archive: memory-safe
streaming parse (line-based, skips `acceleration`/`distance`) → checkpoint Parquet →
cut detection from force AC-RMS (threshold 50 counts) → process detection from
`programrunning` edges → per-cut parameter summary → QA plots. Outputs live in
`data/converted/20260630/` (`lev008_cuts.parquet`, `lev008_processes.parquet`,
`lev008_cut_summary.parquet/csv`, `force_lev008.parquet`, `machine_lev008.parquet`).
Originally pointed at `data/Hermle/20260630/`; that folder has since been renamed to
`data/Hermle/20260630_ALU_y/` (see next notebook).

### `axon_logger_lev008_ALU_y_segmentation_and_nc_log.ipynb`
Extracts a timestamped NC-control command log (start, stop, wait/M00, spindle
near-zero, rapid-traverse/G00 — as edge/state-change events, not a full time series)
from the same lev008 recording, now at `data/Hermle/20260630_ALU_y/`. Confirmed the
file is byte-identical to the one processed above (same `_id`s, time range, block
counts), so it **reuses** the existing cut/process segmentation from
`data/converted/20260630/` instead of re-deriving it, and only computes the new NC
event log, combining it with the reused cuts into a tagged cut summary. Cross-checked
against the actual NC program (`NC_GCode/NUT_CUTO_DA_Y.MPF`) for the spindle-speed/
feed-rate magnitudes used to set thresholds. Outputs in
`data/converted/20260630_ALU_y/` (`nc_command_log_lev008_ALU_y.parquet/csv`,
`lev008_ALU_y_cut_summary_tagged.parquet/csv`, QA plot).

### `axon_logger_20260701_ALU_x_spindle_torque.ipynb`
Computes a measured spindle angular position `phi(t)` from `machine_0`'s absolute
spindle encoder (`axis0_positionact`, hardware-wrapped to `[0, 360)` at 250 Hz — no
RPM integration needed) and projects `force_0`/`force_1` (`Fx`/`Fy`) onto the
tangential direction at that angle to get a torque proxy `Mc(t)`, using the two new
`MillingSignalUtils.phi_from_encoder`/`spindle_torque` methods added to
`py/src/kienzle_utils.py`. Runs against `data/converted/20260701_ALU_x/` (today's
`Lev010-019` data, despite the `*_lev008_ALU_y_raw.parquet` filenames left over from
the copy-pasted conversion notebook). Deliberately does **not** use this same
folder's `lev008_ALU_y_cut_summary_tagged.parquet` / `nc_command_log_lev008_ALU_y.parquet`
— see "Redundancies found" below, they're stale `20260630` data. Outputs
`spindle_torque_qa.png` / `spindle_torque_qa_zoom.png` in the same folder.

### `graphics.ipynb`
Unrelated one-off paper figures: `kc11`/`mc` histograms with fitted Gaussian/Beta
distributions from `data/Kc11_mc.xlsx`, and an actuator-current-vs-time/angle plot
from `data/Trace_0623_164010.csv`. Outputs saved to `nb_aux/results/`.

### `analyse_no _load_data.ipynb`
Unrelated Kienzle-model exploration: fits a no-load current-noise distribution
(`std_iq_no_load` from Sinutrace measurement files) with a Gamma distribution, then
uses `KienzleModel` (`py/src/kienzle_model.py`) to simulate cutting-force traces with
that noise overlaid. Outputs saved to `nb_aux/results/`.

### `results/`
PNG outputs from `graphics.ipynb` and `analyse_no _load_data.ipynb`.

## Redundancies found

- **`axon_logger_json2df_.ipynb` duplicates `axon_logger_json2df.ipynb`.** Every
  markdown and code cell is identical except the default `FileChooser` path/filename
  and the last two cells (downsample/plot), which the `_` copy is missing. Recommend
  deleting `axon_logger_json2df_.ipynb`.

- **The axon-logger parsing helpers are copy-pasted across four notebooks**
  (`axon_logger_json2df.ipynb`, `axon_logger_json2df_.ipynb`,
  `axon_logger_lev008_process_cut_segmentation.ipynb`,
  `axon_logger_lev008_ALU_y_segmentation_and_nc_log.ipynb`) with no shared module —
  `extract_block_meta`, `build_waveform_df`, `_unwrap_number`, `flatten_machine_sample`,
  `build_machine_df` appear verbatim in each. A schema change (e.g. a new axis field)
  has to be patched in up to four places. Worth factoring into a shared module (e.g.
  `py/src/axon_logger_utils.py`) if this format keeps getting new archives.

- **`data/Hermle/20260630/` and `data/Hermle/20260630_ALU_y/` are the same recording**,
  not two experiments — confirmed identical `_id`s, time range, and block counts. The
  two lev008 notebooks therefore overlap in scope; only the NC-command-log section of
  the `ALU_y` notebook is genuinely new work, the cut/process segmentation is reused
  from the older notebook's output rather than recomputed.

- **Stale FileChooser description in both `json2df` notebooks**: the "1 - Pick the
  source JSON file" markdown cell in each says it "starts in `data/Hermle/20260629`
  with `axon_logger_ptw_001.raw.json` pre-selected," but the actual code cell points
  elsewhere in both copies (`20260630/...Lev006...` and `20260629/...00test03...`
  respectively) — leftover text from an earlier version of the notebook.

- **`json2csv.ipynb` is not redundant** with the axon-logger notebooks despite the
  similar name — it handles an entirely different, older MongoDB export schema
  (`*.processed.json` from the `20260623` Kraftmessung/Vary tests, one array per
  record) versus the axon-logger's one-channel-per-record schema.

- **`data/converted/20260701_ALU_x/lev008_ALU_y_cut_summary_tagged.parquet` and
  `nc_command_log_lev008_ALU_y.parquet` are stale.** Whoever adapted
  `axon_logger_lev008_ALU_y_segmentation_and_nc_log.ipynb` for the new
  `20260701_ALU_x` recording kept its `OLD_DIR = .../20260630` reuse of cut/process
  segmentation unchanged — appropriate for the original `lev008_ALU_y` file (proven
  byte-identical to `20260630`), but wrong here since `20260701_ALU_x` is a genuinely
  different recording. The saved `cut_summary_tagged`'s `cut_start`/`cut_end`
  timestamps are literally `2026-06-30`, one day before the `force_0`/`force_1`/
  `machine_0` data in the same folder (which *is* correctly the new 2026-07-01
  recording). Re-run the cut/process-segmentation cells against this file's own data
  before relying on cut windows here.
