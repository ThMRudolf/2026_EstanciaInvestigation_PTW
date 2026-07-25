"""
cnc_cut_extractor.py
---------------------
Datenvorverarbeitungs-Pipeline: axon_logger-MongoDB-JSON-Export ->
ein CSV-File pro Schnitt (Kraft- und Achsdaten), inkl. Winkel-Interpolation
und Radialkraft-Summe an der Schneidkante.

Autor:  Thomas M. Rudolf (ITAM), Pipeline generiert mit Unterstuetzung von Claude
Datum:  2026-07-02

--------------------------------------------------------------------------
VERTRAULICHKEITSHINWEIS
--------------------------------------------------------------------------
Die hier verarbeiteten Rohdaten (axon_logger_*.json) und die daraus erzeugten
Schnitt-CSV-Dateien enthalten Maschinen-, Prozess- und Werkstueckdaten der
PTW-Versuchsreihe (Hermle C32U / SINUMERIK 840D sl, Kistler-Kraftmessplatte).
Diese Klasse selbst nimmt keine Netzwerkverbindung vor -- die gesamte
Verarbeitung laeuft lokal/offline. Beim Weitergeben von Notebook-Outputs
(Plots, CSV-Beispielzeilen) an Dritte bitte pruefen, ob Seriennummern
(serialnr, workpiecedata) oder Klarnamen von Pfaden entfernt werden muessen.
--------------------------------------------------------------------------

Erwartetes JSON-Schema (ein Dokument pro Sensor-Kanal-Sekunde/-Burst):

    {
      "_id": {...},
      "data": [...],                 # Rohdaten, Laenge haengt von sensortype ab
      "device": "...",
      "process": 0 | 1,
      "sensoridx": 0 | 1 | 2,
      "sensortype": "force" | "acceleration" | "distance" | "machine",
      "spindleidx": 0,
      "time": {"$date": "..."},      # Start-Zeitstempel des Dokuments (UTC)
      "workpiecedata": [...]
    }

  - sensortype == "force":        data = 20000 ADC-Rohwerte (int),  20 kHz
  - sensortype == "machine":      data = 250 Snapshots (dict),      250 Hz
        jeder Snapshot enthaelt u.a. "axes": [ {currentact, positionact,
        speedorfeedact, stop, ...}, ... ] (9 Achsen) sowie Status-Flags
        (programrunning, ncstart, ncstop, machineworking, ...).

WICHTIGE, AUS DER STICHPROBE ABGELEITETE ANNAHMEN (bitte an echten Daten
verifizieren und ueber die Konstruktor-Parameter anpassen):

  1. axis[0] ist die Spindel (Position 0-360 deg, Drehzahl ungefaehr
     konstant waehrend eines Schnitts, poweract > 0).
  2. axis[4], axis[5], axis[6] sind die aktiven Vorschubachsen (die drei
     Achsen mit nennenswerter Positionsvarianz); axis[1..3] und axis[7..8]
     sind in der Stichprobe konstant 0 bzw. praktisch unbewegt.
  3. Sowohl "force"- als auch "machine"-Dokumente werden NICHT durchgehend
     geloggt, sondern in Bursts (nur waehrend aktiver Bearbeitung). Ein
     "Block" (Zeitraum zwischen zwei Spindel-/Vorschubstopps, i.d.R. 4
     Schnitte) entspricht einem zusammenhaengenden Force-Burst.
  4. Innerhalb eines Blocks laeuft die Spindel durchgehend; die 4 Schnitte
     werden durch Richtungsumkehr der dominanten Vorschubachse getrennt, 
     bzw. über die z-Position als Ebene (2 x 4 Schnitte sind immer in der
     gleichen z-Position erfolgt) 
     (siehe detect_cuts_in_block).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import ijson
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Dieses Modul benoetigt 'ijson' fuer das speicherschonende "
        "Streaming-Parsing des grossen MongoDB-JSON-Exports. "
        "Installation: pip install ijson --break-system-packages"
    ) from exc

from scipy.signal import find_peaks

from .kienzle_utils import MillingSignalUtils as MSU


# ---------------------------------------------------------------------
# Hilfsfunktion: Mongo-Extended-JSON-Typen (Decimal128, $date, $numberLong)
# rekursiv in native Python-/NumPy-Typen konvertieren
# ---------------------------------------------------------------------
def _to_native(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        if "$numberLong" in value:
            return int(value["$numberLong"])
        if "$numberDouble" in value:
            return float(value["$numberDouble"])
        if "$numberDecimal" in value:
            return float(value["$numberDecimal"])
        if "$date" in value:
            return value["$date"]
        if "$oid" in value:
            return value["$oid"]
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def _naive_ts(value) -> pd.Timestamp:
    """Zeitstempel ohne Zeitzone. Hilfsfunktion fuer die separated_by_ln-Utilities unten, die mit
    absoluten pandas-Timestamps arbeiten (anders als CNCCutExtractor selbst, das intern mit
    Sekunden relativ zu einem globalen t_ref rechnet, siehe _rel_t())."""
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# ---------------------------------------------------------------------
# Eigenstaendige Utility-Funktionen fuer den separated_by_ln-Workflow (ein
# raw.json je bereits korrigierter Laufnummer, siehe laufnummer_correction.py)
# -- unabhaengig von der CNCCutExtractor-Klasse und ihrer block-/reversal-
# basierten Schnitt-Segmentierung. Diese Dateien sind klein genug fuer
# json.load() (anders als die Multi-GB-Rohexporte, fuer die die Klasse oben
# ijson-Streaming benoetigt). Werden u.a. von Abschnitt 17 in
# Schnitt_Extraktion_Beispiel.ipynb importiert: Z-Schwellenwert-
# Eingriffserkennung + X/Z-Stabilitaets-Teilsegmentierung fuer Einzel-Nut-
# Schnitte in Y-Richtung (X und Z muessen waehrend eines echten Schnitts
# nahezu konstant bleiben).
# ---------------------------------------------------------------------

DEFAULT_EXPORT_AXIS_INDICES: Tuple[int, ...] = (0, 4, 5, 6)
DEFAULT_EXPORT_AXIS_FIELDS: Tuple[str, ...] = (
    "currentact", "loadact", "momentumtgt", "positionact", "poweract",
    "speedorfeedact", "speedorfeedovrtgt", "speedorfeedtgt",
)


def load_axes_and_force(
    json_path,
    x_idx: int = 4,
    z_idx: int = 6,
    export_axis_indices: Sequence[int] = DEFAULT_EXPORT_AXIS_INDICES,
    export_axis_fields: Sequence[str] = DEFAULT_EXPORT_AXIS_FIELDS,
) -> Tuple[Optional[pd.DataFrame], Dict[int, pd.DataFrame]]:
    """
    Liest eine separated_by_ln-Datei komplett per json.load und baut (a) die 250-Hz-X/Z-
    Positions-Zeitreihe (Kurznamen x/z, fuer Eingriffs-/Stabilitaets-Segmentierung) plus je Achse
    in export_axis_indices die in export_axis_fields aufgefuehrten Rohwerte (als
    axis{index}_{feld}-Spalten) und (b) je Kraftkanal (sensoridx 0/1/2) eine 20-kHz-Zeitreihe in
    Newton (ADC-Rohwerte via MillingSignalUtils.adc_counts_to_force, bits=16, f_range=1500.0,
    bipolar=True -- gleiche Kalibrierung wie extract_cut()).

    Returns
    -------
    (None, {}) falls die Datei keine 'machine'-Dokumente enthaelt, sonst (pos_df, force_series).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    pos_rows = []
    force_docs: Dict[int, list] = {0: [], 1: [], 2: []}
    for doc in docs:
        sensortype = doc.get("sensortype")
        if sensortype == "machine":
            t0 = _naive_ts(_to_native(doc["time"]))
            for i, snap in enumerate(doc["data"]):
                t = t0 + pd.Timedelta(seconds=i / 250.0)
                row = {
                    "time": t,
                    "x": _to_native(snap["axes"][x_idx]["positionact"]),
                    "z": _to_native(snap["axes"][z_idx]["positionact"]),
                }
                for ai in export_axis_indices:
                    ax = snap["axes"][ai]
                    for field_name in export_axis_fields:
                        row[f"axis{ai}_{field_name}"] = _to_native(ax[field_name])
                pos_rows.append(row)
        elif sensortype == "force":
            idx = doc.get("sensoridx")
            if idx in force_docs:
                t0 = _naive_ts(_to_native(doc["time"]))
                force_docs[idx].append((t0, np.asarray(doc["data"], dtype=float)))

    if not pos_rows:
        return None, {}
    pos_df = pd.DataFrame(pos_rows).sort_values("time").reset_index(drop=True)
    value_cols = [c for c in pos_df.columns if c != "time"]
    for c in value_cols:
        pos_df[c] = pd.to_numeric(pos_df[c], errors="coerce")
    pos_df["time"] = pos_df["time"].astype("datetime64[ns]")

    force_series: Dict[int, pd.DataFrame] = {}
    for idx, docs_list in force_docs.items():
        if not docs_list:
            continue
        docs_list.sort(key=lambda d: d[0])
        t_list, v_list = [], []
        for t0, data in docs_list:
            t_list.append(t0 + pd.to_timedelta(np.arange(len(data)) / 20000.0, unit="s"))
            v_list.append(data)
        t_all = np.concatenate([t.values for t in t_list])
        v_all = np.concatenate(v_list)
        order = np.argsort(t_all)
        df = pd.DataFrame({"time": pd.DatetimeIndex(t_all[order]), "raw": v_all[order]})
        df["time"] = df["time"].astype("datetime64[ns]")
        df["force_n"] = MSU.adc_counts_to_force(df["raw"].to_numpy(), bits=16, f_range=1500.0, bipolar=True)
        force_series[idx] = df

    return pos_df, force_series


def contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Start-/End-Indizes (Ende exklusiv) zusammenhaengender True-Laeufe in einem 1D-Bool-Array."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2], edges[1::2]))


def find_stable_subruns(
    x: np.ndarray, z: np.ndarray, x_thr: float, z_thr: float, min_len: int
) -> List[Tuple[int, int]]:
    """
    Single-Pass GROWING-RUN-Segmentierung (kein festes Rolling-Window): erweitert einen
    Kandidatenlauf sample-fuer-sample, solange sowohl die laufende X- als auch die laufende
    Z-Spannweite unterhalb der jeweiligen Schwelle bleiben (physikalische Begruendung: eine Nut in
    Y-Richtung darf X und Z waehrend eines echten Einzel-Nut-Schnitts nicht nennenswert aendern).
    Sobald eine Erweiterung eine Schwelle ueberschreiten wuerde, wird der aktuelle Lauf geschlossen
    (nur behalten, wenn er >= min_len Samples lang ist) und ein neuer Lauf beim aktuellen Sample
    begonnen. Gibt lokale (start, end)-Indexpaare (Ende exklusiv, relativ zu x/z) zurueck.
    """
    n = len(x)
    if n == 0:
        return []
    subruns = []
    run_start = 0
    x_min = x_max = x[0]
    z_min = z_max = z[0]
    for i in range(1, n):
        new_x_min, new_x_max = min(x_min, x[i]), max(x_max, x[i])
        new_z_min, new_z_max = min(z_min, z[i]), max(z_max, z[i])
        if (new_x_max - new_x_min) < x_thr and (new_z_max - new_z_min) < z_thr:
            x_min, x_max, z_min, z_max = new_x_min, new_x_max, new_z_min, new_z_max
            continue
        if i - run_start >= min_len:
            subruns.append((run_start, i))
        run_start = i
        x_min = x_max = x[i]
        z_min = z_max = z[i]
    if n - run_start >= min_len:
        subruns.append((run_start, n))
    return subruns


def build_segment_table(
    seg_pos: pd.DataFrame,
    seg_force_by_idx: Dict[int, pd.DataFrame],
    t_start: pd.Timestamp,
    merge_tolerance_ms: float = 4,
) -> pd.DataFrame:
    """
    Baut die Export-Tabelle fuer ein Segment (z.B. ein stabiles Teilsegment aus
    find_stable_subruns()): Kraftkanaele (force_0/1/2, 20 kHz, Newton) als Basis-Zeitachse, dazu
    alle axis*_*-Rohwerte aus seg_pos (250 Hz) per merge_asof (nearest) angehaengt -- gleiches
    Prinzip wie raw_table in CNCCutExtractor.extract_cut(). 'time' wird danach zu Sekunden relativ
    zu t_start umgerechnet (Zeitvektor beginnt bei 0).
    """
    table = None
    for idx in sorted(seg_force_by_idx):
        fdf = seg_force_by_idx[idx][["time", "force_n"]].rename(columns={"force_n": f"force_{idx}"})
        table = fdf if table is None else pd.merge_asof(
            table.sort_values("time"), fdf.sort_values("time"), on="time",
            direction="nearest", tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms),
        )
    axis_cols = [c for c in seg_pos.columns if c.startswith("axis")]
    table = pd.merge_asof(
        table.sort_values("time"), seg_pos[["time"] + axis_cols].sort_values("time"), on="time",
        direction="nearest", tolerance=pd.Timedelta(milliseconds=merge_tolerance_ms),
    )
    table["time"] = (table["time"] - t_start).dt.total_seconds()
    return table


@dataclass
class CutResult:
    """Container fuer einen einzelnen extrahierten Schnitt."""

    block_id: int
    cut_id: int
    t_start: float
    t_end: float
    raw_table: pd.DataFrame          # SinuTrace-Stil, Rohsignale
    angle_deg: np.ndarray            # (n_angle_samples,) hochinterpolierte Spindelposition
    force_ds: Dict[str, np.ndarray]  # herunterinterpolierte Kraftkanaele (Newton)
    radial_force_profile: pd.DataFrame  # Summe Fr je Winkel-Bin
    csv_path: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class CNCCutExtractor:
    """
    Vorverarbeitungs-Pipeline: liest den axon_logger-MongoDB-JSON-Export,
    segmentiert die Kraft-/Achsdaten in einzelne Schnitte und schreibt pro
    Schnitt ein CSV-File (Signalnamen wie im JSON / wie im SinuTrace-
    Referenz-Export). Zusaetzlich wird pro Schnitt ein Winkel-aufgeloestes
    Radialkraft-Profil an der Schneidkante berechnet.

    Parameter
    ---------
    json_path : str
        Pfad zum axon_logger-*.json-Export (MongoDB-Extended-JSON, Array
        von Dokumenten).
    output_dir : str
        Zielverzeichnis fuer die Schnitt-CSV-Dateien (wird angelegt, falls
        nicht vorhanden).
    spindle_axis_index : int, default 0
        Index der Spindelachse im 'axes'-Array der 'machine'-Dokumente.
    feed_axis_indices : Sequence[int], default (4, 5, 6)
        Indizes der zu exportierenden/auszuwertenden Vorschubachsen.
    cut_axis_index : int
        Vorschubachse, die zur Schnitt-Segmentierung (Richtungsumkehr-
        Erkennung) verwendet wird -- muss explizit gesetzt werden (keine
        automatische Achsauswahl anhand der Positions-Spannweite mehr,
        da diese bei reinen Y-/X-/Z-Schnitten faelschlich auf Rauschen/
        Restbewegung einer anderen Achse anspringen konnte). Muss in
        feed_axis_indices enthalten sein.
    cut_direction_sign : int, optional
        Erzwingt zusaetzlich zu cut_axis_index (muss dafuer gesetzt sein)
        die Bewegungsrichtung, die als echter Schnitt gilt: +1 = Position
        steigt, -1 = Position faellt. Segmente in die andere Richtung
        werden verworfen (Ruecklauf/Leerfahrt). Default None (keine
        Richtungs-Einschraenkung). GRUND: bei manchen Auftraegen laeuft
        der eigentliche Schnitt immer in eine feste Richtung (z.B. immer
        -X oder -Y); kombiniert mit max_cut_velocity ergibt sich eine
        praezisere Filterung als Geschwindigkeit allein. Vor Aktivierung
        immer zuerst cut_diagnostics() im Notebook pruefen (direction_sign-
        Spalte) -- die Konvention ist datensatz-/setup-abhaengig.
    cuts_per_block : int, default 4
        Erwartete Anzahl Schnitte je Block (nur fuer Plausibilitaets-
        Warnung genutzt, kein hartes Limit).
    n_angle_samples : int, default 1000
        Ziel-Abtastzahl fuer die hochinterpolierte Spindelposition
        (0-360 deg) je Schnitt.
    n_force_samples : int, default 1000
        Ziel-Abtastzahl fuer die herunterinterpolierten Kraftkanaele je
        Schnitt (muss mit n_angle_samples uebereinstimmen, damit Winkel
        und Kraft punktweise zusammenpassen).
    adc_bits : int, default 16
        Aufloesung des Kraft-ADC (siehe MSU.adc_counts_to_force).
    adc_f_range : float, default 1500.0
        Konfigurierter Kraftbereich des Kistler-Ladungsverstaerkers in
        Newton (siehe MSU.adc_counts_to_force).
    adc_bipolar : bool, default True
        Ob der ADC bipolar (+/-10 V oder +/- 1500N) oder unipolar (0-10 V oder 0 - 1500 N) ist.
    min_block_duration_s : float, default 3.0
        Minimale Dauer eines zusammenhaengenden Force-Bursts, damit er als
        "Bearbeitungsblock" (und nicht als Leerlauf-Heartbeat-Sample)
        gewertet wird.
    min_burst_amplitude : float, optional
        Minimale Peak-to-Peak-Amplitude (rohe ADC-Counts, Maximum ueber
        alle FORCE_SENSOR_INDICES-Kanaele) eines Bursts, damit dieser
        ueberhaupt als moeglicher Bearbeitungsblock in Frage kommt.
        Default None (deaktiviert -- detect_blocks() verhaelt sich dann
        exakt wie bisher, rein dauerbasiert). GRUND: in einer Stichprobe
        (Lev022) gab es sowohl lange (>= min_block_duration_s) Bursts auf
        Rauschniveau (~9 Counts, programrunning=False) als auch kurze
        1s-"Heartbeat"-Bursts mit eindeutigem Zerspankraftniveau
        (4000-7000 Counts, programrunning=True). Der Nutzen ist
        datensatzabhaengig (in einer zweiten Stichprobe war das
        Kraftsignal durchgehend zu schwach) -- daher vor Aktivierung immer
        zuerst burst_diagnostics() im Notebook pruefen.
    min_programrunning_frac : float, default 0.5
        Nur relevant, wenn min_burst_amplitude gesetzt ist. Mindestanteil
        (0-1) der 'machine'-Snapshots im Burst-Fenster mit
        programrunning == True, damit ein Burst trotz Dauer <
        min_block_duration_s als Block "gerettet" wird.
    reversal_smooth_window : int, default 15
        Fensterbreite (Anzahl 250-Hz-Samples) der Glaettung vor der
        Richtungsumkehr-Erkennung der Vorschubachse.
    reversal_prominence_frac : float, default 0.1
        Mindest-Prominenz eines Richtungsumkehr-Peaks, als Anteil der
        Positions-Spannweite der dominanten Vorschubachse im Block.
    max_cut_velocity : float, optional
        Maximale mittlere Geschwindigkeit (Achsposition/s) der dominanten
        Vorschubachse ueber ein rohes Segment, damit dieses noch als
        echter (materialabtragender) Schnitt gilt. Segmente, deren
        mittlere Geschwindigkeit |Delta Position| / Dauer diese Schwelle
        ueberschreitet, werden als nicht-schneidender Ruecklauf/Leerfahrt
        verworfen. Default None (deaktiviert -- detect_cuts_in_block()
        verhaelt sich dann exakt wie bisher). GRUND: in einer Stichprobe
        (Lev022, dominante Achse axis5/Y) hatten echte Schnitte durchweg
        niedrige mittlere Geschwindigkeit (~7-10 Einheiten/s, Dauer 2.8-
        11s) und Rueckhuebe durchweg hohe (~36-63 Einheiten/s, Dauer
        < 2.8s) -- Groessenordnungs-Luecke. Die Bewegungsrichtung selbst
        ist NICHT Teil der Filterlogik (datensatzabhaengig, siehe
        cut_diagnostics()); nur der Geschwindigkeitsbetrag wird
        ausgewertet. Vor Aktivierung immer zuerst cut_diagnostics() im
        Notebook pruefen (Schwelle ist datensatzabhaengig).
    max_other_axis_range : float, optional
        Maximale erlaubte Positions-Spannweite (gleiche Einheit wie
        positionact, i.d.R. mm) der NICHT gewaehlten Vorschubachsen
        (feed_axis_indices ohne cut_axis_index) waehrend eines Schnitt-
        Segments. GRUND: ein echter Einachs-Schnitt (z.B. in X) soll die
        jeweils anderen Achsen (Y, Z) nur im Rahmen von Rauschen/
        Vibration bewegen -- bewegt sich eine andere Achse waehrend eines
        Segments ueber diese Schwelle hinaus, ist es vermutlich kein
        reiner Einachs-Schnitt (z.B. gleichzeitige Y-Verfahrbewegung) und
        wird als Ruecklauf/Leerfahrt verworfen. Default None (deaktiviert
        -- detect_cuts_in_block() verhaelt sich dann exakt wie bisher).
        Vor Aktivierung immer zuerst cut_diagnostics() im Notebook
        pruefen (other_axis_max_range-Spalte) -- die Schwelle ist
        rauschabhaengig.
    r_tool_mm : float, optional
        Werkzeugradius in mm fuer die Radialkraft-Berechnung. Falls None,
        wird versucht, ihn aus dem 'toolradius'-Feld der 'machine'-
        Dokumente zu lesen.
    """

    FORCE_SENSOR_INDICES = (0, 1, 2)

    def __init__(
        self,
        json_path: str,
        output_dir: str = "./schnitte",
        spindle_axis_index: int = 0,
        feed_axis_indices: Sequence[int] = (4, 5, 6),
        cut_axis_index: Optional[int] = None,  # Pflichtparameter -- siehe Validierung unten
        cut_direction_sign: Optional[int] = None,
        cuts_per_block: int = 4,
        n_angle_samples: int = 1000,
        n_force_samples: int = 1000,
        adc_bits: int = 16,
        adc_f_range: float = 1500.0,
        adc_bipolar: bool = True,
        min_block_duration_s: float = 3.0,
        min_burst_amplitude: Optional[float] = None,
        min_programrunning_frac: float = 0.5,
        reversal_smooth_window: int = 15,
        reversal_prominence_frac: float = 0.1,
        min_feed_range: float = 2.0,
        max_cut_velocity: Optional[float] = None,
        max_other_axis_range: Optional[float] = None,
        r_tool_mm: Optional[float] = None,
    ):
        self.json_path = json_path
        self.output_dir = output_dir
        self.spindle_axis_index = spindle_axis_index
        self.feed_axis_indices = tuple(feed_axis_indices)
        if cut_axis_index is None:
            raise ValueError(
                "cut_axis_index muss explizit gesetzt werden -- die Schnitt-Segmentierung "
                f"verwendet ausschliesslich die vorgegebene Achse (keine automatische Auswahl "
                f"anhand der Positions-Spannweite mehr). Gueltige Werte: {tuple(feed_axis_indices)}."
            )
        if cut_axis_index not in self.feed_axis_indices:
            raise ValueError(
                f"cut_axis_index={cut_axis_index} ist nicht in feed_axis_indices="
                f"{self.feed_axis_indices} enthalten."
            )
        self.cut_axis_index = cut_axis_index
        if cut_direction_sign is not None:
            if cut_axis_index is None:
                raise ValueError(
                    "cut_direction_sign gesetzt, aber cut_axis_index ist None -- "
                    "Richtung ist ohne explizit vorgegebene Achse nicht eindeutig."
                )
            if cut_direction_sign not in (-1, 1):
                raise ValueError(
                    f"cut_direction_sign={cut_direction_sign} muss -1, +1 oder None sein."
                )
        self.cut_direction_sign = cut_direction_sign
        self.cuts_per_block = cuts_per_block
        self.n_angle_samples = n_angle_samples
        self.n_force_samples = n_force_samples
        self.adc_bits = adc_bits
        self.adc_f_range = adc_f_range
        self.adc_bipolar = adc_bipolar
        self.min_block_duration_s = min_block_duration_s
        if not (0.0 <= min_programrunning_frac <= 1.0):
            raise ValueError(
                f"min_programrunning_frac={min_programrunning_frac} muss in [0, 1] liegen."
            )
        self.min_burst_amplitude = min_burst_amplitude
        self.min_programrunning_frac = min_programrunning_frac
        self.reversal_smooth_window = reversal_smooth_window
        self.reversal_prominence_frac = reversal_prominence_frac
        self.min_feed_range = min_feed_range
        if max_cut_velocity is not None and max_cut_velocity <= 0:
            raise ValueError(
                f"max_cut_velocity={max_cut_velocity} muss > 0 sein (oder None)."
            )
        self.max_cut_velocity = max_cut_velocity
        if max_other_axis_range is not None and max_other_axis_range < 0:
            raise ValueError(
                f"max_other_axis_range={max_other_axis_range} muss >= 0 sein (oder None)."
            )
        self.max_other_axis_range = max_other_axis_range
        self.r_tool_mm = r_tool_mm

        os.makedirs(self.output_dir, exist_ok=True)

        # werden in load() befuellt
        self._force_docs: Dict[int, List[dict]] = {0: [], 1: [], 2: []}
        self._machine_docs: List[dict] = []
        self._t_ref: Optional[pd.Timestamp] = None  # globaler Zeit-Nullpunkt

        self.results: List[CutResult] = []
        self.summary_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # 1. Laden / Streaming-Parsing des JSON
    # ------------------------------------------------------------------
    def load(self) -> "CNCCutExtractor":
        """
        Parst den JSON-Export einmalig per ijson-Stream (speicherschonend
        fuer 300+ MB-Dateien) und behaelt nur die fuer die Pipeline
        benoetigten sensortype-Kanaele ("force", "machine") im Speicher.
        "acceleration" und "distance" werden ignoriert (nicht Teil der
        Aufgabenstellung).
        """
        times_seen: List[pd.Timestamp] = []

        with open(self.json_path, "rb") as f:
            for doc in ijson.items(f, "item"):
                sensortype = doc["sensortype"]
                if sensortype not in ("force", "machine"):
                    continue

                t0 = pd.Timestamp(doc["time"]["$date"])
                times_seen.append(t0)

                if sensortype == "force":
                    idx = doc["sensoridx"]
                    self._force_docs[idx].append(
                        {
                            "t0": t0,
                            "n": len(doc["data"]),
                            "data": np.asarray(doc["data"], dtype=np.float64),
                            "process": doc["process"],
                        }
                    )
                else:  # machine
                    self._machine_docs.append(
                        {
                            "t0": t0,
                            "data": doc["data"],  # bleibt Liste von dicts (Decimal etc.)
                            "process": doc["process"],
                        }
                    )

        if not times_seen:
            raise ValueError(
                f"Keine 'force'- oder 'machine'-Dokumente in {self.json_path} gefunden."
            )
        self._t_ref = min(times_seen)

        for idx in self._force_docs:
            self._force_docs[idx].sort(key=lambda d: d["t0"])
        self._machine_docs.sort(key=lambda d: d["t0"])

        print(
            f"[load] force-Dokumente: "
            f"{ {k: len(v) for k, v in self._force_docs.items()} }, "
            f"machine-Dokumente: {len(self._machine_docs)}, "
            f"t_ref={self._t_ref}"
        )
        return self

    def _rel_t(self, ts: pd.Timestamp) -> float:
        return (ts - self._t_ref).total_seconds()

    # ------------------------------------------------------------------
    # 2. Kraft-Bursts (zusammenhaengende Aufnahme-Fenster) bestimmen
    # ------------------------------------------------------------------
    def _force_bursts(self) -> List[dict]:
        """
        Gruppiert die force_0-Dokumente (als Referenzkanal) in Bursts:
        aufeinanderfolgende Dokumente gelten als zusammenhaengend, wenn das
        naechste Dokument exakt dort beginnt, wo das vorherige endet
        (Toleranz 10 ms). Ein Burst = potentieller "Bearbeitungsblock".
        """
        ref_docs = self._force_docs[0]
        if not ref_docs:
            raise ValueError("Keine force_0-Dokumente geladen -- load() zuerst aufrufen.")

        bursts = []
        current = None
        for d in ref_docs:
            t_start = self._rel_t(d["t0"])
            dur = d["n"] / 20000.0
            t_end = t_start + dur
            if current is None or (t_start - current["t_end"]) > 0.01:
                if current is not None:
                    bursts.append(current)
                current = {"t_start": t_start, "t_end": t_end, "docs": [d]}
            else:
                current["t_end"] = t_end
                current["docs"].append(d)
        if current is not None:
            bursts.append(current)
        return bursts

    def _concat_force_channel(self, sensoridx: int, t_start: float, t_end: float
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Baut fuer einen gegebenen Zeitfenster [t_start, t_end] die
        konkatenierte (t, raw_counts)-Zeitreihe eines einzelnen
        Kraft-Kanals (sensoridx) aus den geladenen Dokumenten auf.
        """
        docs = self._force_docs[sensoridx]
        t_list, v_list = [], []
        for d in docs:
            d_t0 = self._rel_t(d["t0"])
            d_dur = d["n"] / 20000.0
            d_t1 = d_t0 + d_dur
            if d_t1 < t_start - 0.01 or d_t0 > t_end + 0.01:
                continue
            t_axis = d_t0 + np.arange(d["n"]) / 20000.0
            t_list.append(t_axis)
            v_list.append(d["data"])
        if not t_list:
            return np.array([]), np.array([])
        t = np.concatenate(t_list)
        v = np.concatenate(v_list)
        order = np.argsort(t)
        return t[order], v[order]

    # ------------------------------------------------------------------
    # 3. Machine-Channel: Achsdaten (Spindel + Vorschub) als DataFrame
    # ------------------------------------------------------------------
    def _machine_frame(self, t_start: float, t_end: float) -> pd.DataFrame:
        """
        Baut aus den 'machine'-Dokumenten, die das Fenster [t_start, t_end]
        (mit etwas Rand) ueberlappen, einen 250-Hz-DataFrame mit Spindel-
        und Vorschubachsen-Signalen sowie den GCode-Status-Flags.
        """
        axes_of_interest = (self.spindle_axis_index,) + self.feed_axis_indices
        rows = []
        for d in self._machine_docs:
            d_t0 = self._rel_t(d["t0"])
            n = len(d["data"])
            d_t1 = d_t0 + n / 250.0
            if d_t1 < t_start - 0.5 or d_t0 > t_end + 0.5:
                continue
            for i, snap in enumerate(d["data"]):
                t = d_t0 + i / 250.0
                axes = snap["axes"]
                row = {
                    "t": t,
                    "programrunning": snap["programrunning"],
                    "ncstart": snap["ncstart"],
                    "ncstop": snap["ncstop"],
                    "machineworking": snap["machineworking"],
                    "programstopped": snap["programstopped"],
                    "reset": snap["reset"],
                }
                for ai in axes_of_interest:
                    ax = axes[ai]
                    row[f"axis{ai}_currentact"] = _to_native(ax["currentact"])
                    row[f"axis{ai}_positionact"] = _to_native(ax["positionact"])
                    row[f"axis{ai}_stop"] = ax["stop"]
                rows.append(row)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        # Erzwingt float-Spalten fuer die Achswerte, unabhaengig davon, ob im JSON ueberall
        # native floats oder (teilweise) Mongo-Extended-JSON-Zahlentypen vorlagen -- verhindert,
        # dass beim CSV-Export einzelne Zellen als Text-/Objekt-Repraesentation landen.
        for ai in axes_of_interest:
            df[f"axis{ai}_currentact"] = pd.to_numeric(df[f"axis{ai}_currentact"], errors="coerce")
            df[f"axis{ai}_positionact"] = pd.to_numeric(df[f"axis{ai}_positionact"], errors="coerce")
        return df

    def machine_frame(self, t_start: float, t_end: float) -> pd.DataFrame:
        """Oeffentlicher Zugriff auf _machine_frame() fuer Diagnose-Plots im Notebook."""
        return self._machine_frame(t_start, t_end)

    def concat_force_channel(self, sensoridx: int, t_start: float, t_end: float
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Oeffentlicher Zugriff auf _concat_force_channel() fuer Diagnose-Plots im Notebook."""
        return self._concat_force_channel(sensoridx, t_start, t_end)

    def data_time_range(self) -> Tuple[float, float]:
        """
        Gesamter Zeitbereich (relativ zu t_ref, in Sekunden) ueber alle geladenen
        force- und machine-Dokumente hinweg -- unabhaengig von der Block-/Schnitt-
        Erkennung (die kurze Heartbeat-Bursts am Rand verwirft). Nuetzlich, um z.B.
        "die letzten N Sekunden der Datei" ohne Segmentierung zu plotten.
        """
        end_times = []
        for docs in self._force_docs.values():
            for d in docs:
                end_times.append(self._rel_t(d["t0"]) + d["n"] / 20000.0)
        for d in self._machine_docs:
            end_times.append(self._rel_t(d["t0"]) + len(d["data"]) / 250.0)
        if not end_times:
            raise ValueError("Keine Dokumente geladen -- load() zuerst aufrufen.")
        return 0.0, max(end_times)

    def _burst_amplitude(self, t_start: float, t_end: float) -> float:
        """
        Maximale Peak-to-Peak-Amplitude (rohe ADC-Counts) ueber alle
        FORCE_SENSOR_INDICES-Kanaele im Fenster [t_start, t_end]. 0.0,
        falls auf keinem Kanal Daten im Fenster liegen.
        """
        amps = []
        for si in self.FORCE_SENSOR_INDICES:
            t, raw = self._concat_force_channel(si, t_start, t_end)
            if t.size == 0:
                continue
            mask = (t >= t_start) & (t <= t_end)
            raw = raw[mask]
            if raw.size == 0:
                continue
            amps.append(float(np.ptp(raw)))
        return max(amps) if amps else 0.0

    def _programrunning_frac(self, t_start: float, t_end: float) -> float:
        """
        Anteil (0-1) der 'machine'-Snapshots im Fenster [t_start, t_end]
        mit programrunning == True. _machine_frame() puffert bis zu 0.5s
        ueber [t_start, t_end] hinaus (dokumentweise, nicht snapshot-
        genau) -- daher hier explizit auf das exakte Fenster maskieren,
        sonst wuerde bei kurzen (~1s) Heartbeat-Bursts der
        Nachbarblock/-leerlauf das Ergebnis dominieren (analog zur
        Maskierung in extract_cut()). 0.0, falls keine 'machine'-Daten im
        Fenster liegen.
        """
        mdf = self._machine_frame(t_start, t_end)
        if mdf.empty:
            return 0.0
        mdf = mdf[(mdf["t"] >= t_start) & (mdf["t"] <= t_end)]
        if mdf.empty:
            return 0.0
        return float(mdf["programrunning"].mean())

    # ------------------------------------------------------------------
    # 4. Bloecke bestimmen (Bearbeitungsfenster zwischen Spindel-/
    #    Vorschubstopps)
    # ------------------------------------------------------------------
    def detect_blocks(self) -> List[dict]:
        """
        Liefert alle Force-Bursts, deren Dauer >= min_block_duration_s ist
        (kurze 1s-"Heartbeat"-Bursts ohne echten Bearbeitungseingriff
        werden verworfen). Jeder verbleibende Burst entspricht einem Block
        von (i.d.R.) cuts_per_block Schnitten.

        Falls min_burst_amplitude gesetzt ist (siehe Konstruktor-Doku),
        wird zusaetzlich Amplitude UND programrunning-Anteil herangezogen:
        ein Burst wird gehalten, wenn
            amplitude >= min_burst_amplitude
            UND (duration >= min_block_duration_s
                 ODER programrunning_frac >= min_programrunning_frac).
        Mit min_burst_amplitude=None (Default) ist das Verhalten
        byte-identisch zur reinen Dauer-Heuristik.
        """
        bursts = self._force_bursts()

        if self.min_burst_amplitude is None:
            blocks = [b for b in bursts if (b["t_end"] - b["t_start"]) >= self.min_block_duration_s]
            print(
                f"[detect_blocks] {len(bursts)} Force-Bursts gefunden, "
                f"{len(blocks)} davon >= {self.min_block_duration_s}s "
                f"(als Bearbeitungsbloecke gewertet)."
            )
            return blocks

        blocks = []
        n_rejected_noise = 0
        n_recovered_short = 0
        for b in bursts:
            dur = b["t_end"] - b["t_start"]
            duration_ok = dur >= self.min_block_duration_s
            amp_ok = self._burst_amplitude(b["t_start"], b["t_end"]) >= self.min_burst_amplitude
            if not amp_ok:
                if duration_ok:
                    n_rejected_noise += 1
                continue
            if duration_ok:
                blocks.append(b)
                continue
            if self._programrunning_frac(b["t_start"], b["t_end"]) >= self.min_programrunning_frac:
                blocks.append(b)
                n_recovered_short += 1
        print(
            f"[detect_blocks] {len(bursts)} Force-Bursts gefunden, {len(blocks)} als "
            f"Bearbeitungsbloecke gewertet (kombinierte Amplitude+Dauer/programrunning-"
            f"Logik, min_burst_amplitude={self.min_burst_amplitude}): "
            f"{n_rejected_noise} lange Bursts als Rauschen verworfen, "
            f"{n_recovered_short} kurze Bursts als echte Bloecke gerettet."
        )
        return blocks

    def burst_diagnostics(self) -> pd.DataFrame:
        """
        Diagnose-Tabelle ueber ALLE Force-Bursts (nicht nur die aktuell
        als Block gewerteten): Dauer, maximale Peak-to-Peak-Amplitude und
        programrunning-Anteil je Burst, sowie ob der Burst unter der
        reinen Dauer-Logik bzw. der kombinierten Logik (falls
        min_burst_amplitude gesetzt waere) als Block gewertet wuerde.

        Gedacht zum Vor-Pruefen im Notebook (Abschnitt 2), BEVOR
        min_burst_amplitude/min_programrunning_frac scharf geschaltet
        werden -- analog zum "visuell pruefen"-Vorgehen fuer die
        Schnitt-Segmentierung in Abschnitt 3.
        """
        bursts = self._force_bursts()
        rows = []
        for i, b in enumerate(bursts, start=1):
            t0, t1 = b["t_start"], b["t_end"]
            dur = t1 - t0
            amp = self._burst_amplitude(t0, t1)
            pr_frac = self._programrunning_frac(t0, t1)
            kept_duration_only = dur >= self.min_block_duration_s
            if self.min_burst_amplitude is None:
                kept_combined = kept_duration_only
            else:
                kept_combined = (amp >= self.min_burst_amplitude) and (
                    kept_duration_only or pr_frac >= self.min_programrunning_frac
                )
            rows.append({
                "burst_id": i, "t_start": t0, "t_end": t1, "duration_s": dur,
                "amplitude_ptp": amp, "programrunning_frac": pr_frac,
                "kept_duration_only": kept_duration_only,
                "kept_combined": kept_combined,
            })
        return pd.DataFrame(rows)

    def _sampling_channels(self) -> Dict[str, Tuple[List[dict], float, Callable[[dict], int]]]:
        """Kanaele fuer sampling_diagnostics()/sampling_diagnostics_in_windows(): (docs, Hz, n(doc))."""
        channels: Dict[str, Tuple[List[dict], float, Callable[[dict], int]]] = {
            "machine": (self._machine_docs, 250.0, lambda d: len(d["data"])),
        }
        for idx, docs in self._force_docs.items():
            channels[f"force_{idx}"] = (docs, 20000.0, lambda d: d["n"])
        return channels

    def sampling_diagnostics(self) -> pd.DataFrame:
        """
        Prueft je Kanal (machine + force_0/1/2), ob die per load() geladenen
        Rohdokumente eine luecken lose Zeitreihe bilden oder in einzelnen
        Aufnahme-Bursts (getrennt durch Idle-Luecken) vorliegen -- unabhaengig
        von der Block-/Schnitt-Erkennung. Toleranz fuer eine "Luecke" ist das
        2-fache der (kanalspezifischen) medianen Dokumentdauer, um Rundungs-/
        Jitter-Effekte zu tolerieren.

        Gedacht, um zu vergleichen, ob Positions- (250Hz) und Kraftkanaele
        (20kHz) gleichermassen luecken behaftet sind (sie sind es -- beide
        werden vom axon_logger nur in 1s-Bursts geschrieben, siehe
        sampling_diagnostics_in_windows() fuer den Nachweis, dass innerhalb
        eines echten Bearbeitungsblocks keine Luecken auftreten).
        """
        rows = []
        for name, (docs, rate, n_fn) in self._sampling_channels().items():
            if not docs:
                continue
            order = sorted(range(len(docs)), key=lambda i: docs[i]["t0"])
            t0s = np.array([self._rel_t(docs[i]["t0"]) for i in order])
            durs = np.array([n_fn(docs[i]) / rate for i in order])
            t1s = t0s + durs
            gaps = t0s[1:] - t1s[:-1] if len(docs) > 1 else np.array([])
            tol = 2.0 * float(np.median(durs))
            span = float(t1s[-1] - t0s[0]) if len(docs) else 0.0
            rows.append({
                "channel": name, "n_docs": len(docs), "covered_s": float(durs.sum()),
                "span_s": span,
                "coverage_frac": float(durs.sum() / span) if span > 0 else 1.0,
                "n_gaps": int((gaps > tol).sum()),
                "max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
                "mean_gap_s": float(gaps.mean()) if len(gaps) else 0.0,
            })
        return pd.DataFrame(rows)

    def sampling_diagnostics_in_windows(self, windows: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        Wie sampling_diagnostics(), aber je Fenster aus `windows` (z.B. die
        Bloecke aus detect_blocks()) statt ueber die gesamte Aufzeichnung:
        Luecken werden nur zwischen aufeinanderfolgenden Dokumenten INNERHALB
        desselben Fensters gezaehlt (nicht zwischen zwei Bloecken). Damit laesst
        sich nachweisen, dass echte Bearbeitungsbloecke -- anders als die
        Aufzeichnung als Ganzes -- luecken los sind (fully_continuous == True).
        """
        rows = []
        for w_id, (t_start, t_end) in enumerate(windows, start=1):
            for name, (docs, rate, n_fn) in self._sampling_channels().items():
                t0s_all = np.array([self._rel_t(d["t0"]) for d in docs])
                durs_all = np.array([n_fn(d) / rate for d in docs])
                t1s_all = t0s_all + durs_all
                mask = (t0s_all >= t_start - 0.01) & (t1s_all <= t_end + 0.01)
                idxs = np.argsort(t0s_all[mask])
                t0s, t1s = t0s_all[mask][idxs], t1s_all[mask][idxs]
                if len(t0s) == 0:
                    rows.append({
                        "window_id": w_id, "t_start": t_start, "t_end": t_end,
                        "channel": name, "n_docs": 0, "coverage_frac": 0.0,
                        "max_gap_s": float("nan"), "fully_continuous": False,
                    })
                    continue
                gaps = t0s[1:] - t1s[:-1] if len(t0s) > 1 else np.array([0.0])
                tol = 2.0 * float(np.median(durs_all[mask][idxs]))
                span = t_end - t_start
                rows.append({
                    "window_id": w_id, "t_start": t_start, "t_end": t_end,
                    "channel": name, "n_docs": int(len(t0s)),
                    "coverage_frac": float((t1s - t0s).sum() / span) if span > 0 else 1.0,
                    "max_gap_s": float(gaps.max()),
                    "fully_continuous": bool(gaps.max() <= tol),
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 5. Einzelne Schnitte innerhalb eines Blocks trennen
    #    (Vorschubachsen-Richtungsumkehr)
    # ------------------------------------------------------------------
    def _cut_segments_raw(
        self, block: dict
    ) -> Tuple[List[Tuple[float, float]], List[str], Optional[int], Optional[float],
               Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[int, np.ndarray]]]:
        """
        Reine Segmentierungs-Logik (Richtungsumkehr-Erkennung + Kurzsegment-
        Merge), OHNE max_cut_velocity-/max_other_axis_range-Filterung und
        OHNE die "N Schnitte erkannt"-Plausibilitaets-Warnung (die haengt
        vom tatsaechlich zurueckgegebenen -- ggf. gefilterten -- cuts ab
        und wird daher separat in detect_cuts_in_block() ausgewertet).
        Wird von detect_cuts_in_block() UND cut_diagnostics() verwendet,
        damit die Segmentierungslogik nur an einer Stelle existiert.

        Returns
        -------
        cuts, msgs, dom_axis, dom_axis_range, t, pos, other_axes_pos --
        die letzten fuenf sind None in den beiden Fallback-Faellen (keine
        'machine'-Daten im Blockfenster / Positions-Spannweite <
        min_feed_range), da dort kein verlaessliches Achssignal fuer eine
        Geschwindigkeits-/Nebenachsen-Berechnung vorliegt. other_axes_pos
        ist ein Dict {axis_index: positionact-Array (auf t ausgerichtet)}
        fuer alle feed_axis_indices ausser dom_axis -- Basis fuer die
        max_other_axis_range-Filterung/-Diagnose.
        """
        msgs = []
        t_start, t_end = block["t_start"], block["t_end"]
        mdf = self._machine_frame(t_start, t_end)

        if mdf.empty:
            msgs.append(
                "Keine 'machine'-Daten im Blockfenster gefunden -- "
                f"Fallback: gleichmaessige Teilung in {self.cuts_per_block} Schnitte."
            )
            edges = np.linspace(t_start, t_end, self.cuts_per_block + 1)
            return list(zip(edges[:-1], edges[1:])), msgs, None, None, None, None, None

        # dominante Vorschubachse = ausschliesslich die per cut_axis_index
        # vorgegebene Achse (Pflichtparameter, keine automatische Auswahl
        # anhand der Positions-Spannweite mehr).
        dom_axis = self.cut_axis_index
        pos = mdf[f"axis{dom_axis}_positionact"].values
        t = mdf["t"].values
        axis_range = float(pos.max() - pos.min()) if pos.size else 0.0
        other_axes_pos = {
            ai: mdf[f"axis{ai}_positionact"].values
            for ai in self.feed_axis_indices if ai != dom_axis
        }

        if axis_range < self.min_feed_range:
            msgs.append(
                f"Vorgegebene Schnittachse axis{dom_axis} bewegt sich um weniger als "
                f"min_feed_range={self.min_feed_range} (Spannweite: {axis_range:.3f}) -- "
                f"vermutlich reiner Spindel-Leerlauf oder zu kurzes/verrauschtes "
                f"Machine-Datenfenster. Fallback: gleichmaessige Teilung in "
                f"{self.cuts_per_block} Schnitte."
            )
            edges = np.linspace(t_start, t_end, self.cuts_per_block + 1)
            return list(zip(edges[:-1], edges[1:])), msgs, None, None, None, None, None

        pos_smooth = (
            pd.Series(pos)
            .rolling(self.reversal_smooth_window, center=True, min_periods=1)
            .mean()
            .values
        )
        prom = axis_range * self.reversal_prominence_frac
        peaks_max, _ = find_peaks(pos_smooth, prominence=prom)
        peaks_min, _ = find_peaks(-pos_smooth, prominence=prom)
        reversal_idx = np.sort(np.concatenate([peaks_max, peaks_min])).astype(int)
        reversal_t = t[reversal_idx].tolist()

        edges = [t_start] + reversal_t + [t_end]
        edges = sorted(set(edges))
        cuts = list(zip(edges[:-1], edges[1:]))
        # sehr kurze Rest-Segmente (< 10% der mittleren Schnittdauer) an
        # den Vorgaenger anhaengen (typ. Restrampe am Blockrand)
        if len(cuts) > 1:
            mean_dur = np.mean([c[1] - c[0] for c in cuts])
            cleaned = [cuts[0]]
            for c in cuts[1:]:
                if (c[1] - c[0]) < 0.1 * mean_dur:
                    cleaned[-1] = (cleaned[-1][0], c[1])
                else:
                    cleaned.append(c)
            cuts = cleaned

        return cuts, msgs, dom_axis, axis_range, t, pos, other_axes_pos

    def detect_cuts_in_block(self, block: dict) -> Tuple[List[Tuple[float, float]], List[str]]:
        """
        Erkennt die Grenzen der einzelnen Schnitte innerhalb eines Blocks
        anhand der Richtungsumkehr des Positions-Signals der per
        cut_axis_index vorgegebenen Vorschubachse (einziges Kriterium fuer
        die Schnittachse, keine automatische Auswahl).

        Fallback: Wenn keine 'machine'-Daten das Blockfenster ueberlappen
        oder keine Richtungsumkehr erkannt wird, wird der Block in
        cuts_per_block gleich lange Zeitabschnitte geteilt (mit Warnung).

        Falls max_cut_velocity, cut_direction_sign und/oder
        max_other_axis_range gesetzt sind (siehe Konstruktor-Doku), werden
        rohe Segmente mit zu hoher mittlerer Geschwindigkeit, falscher
        Bewegungsrichtung bzw. zu grosser Mitbewegung einer NICHT
        gewaehlten Vorschubachse zusaetzlich als Ruecklauf/Leerfahrt
        verworfen -- vorher mit cut_diagnostics() pruefen.

        Returns
        -------
        cuts : list of (t_start, t_end)
        msgs : list of Warnhinweise (fuer Protokollierung)
        """
        cuts, msgs, dom_axis, dom_axis_range, t, pos, other_axes_pos = self._cut_segments_raw(block)

        any_filter_active = (
            self.max_cut_velocity is not None
            or self.cut_direction_sign is not None
            or self.max_other_axis_range is not None
        )
        if any_filter_active and dom_axis is not None:
            kept: List[Tuple[float, float]] = []
            n_filtered = 0
            for t0, t1 in cuts:
                dur = t1 - t0
                if dur <= 0:
                    kept.append((t0, t1))
                    continue
                p0 = np.interp(t0, t, pos)
                p1 = np.interp(t1, t, pos)
                vel = abs(p1 - p0) / dur
                vel_ok = self.max_cut_velocity is None or vel <= self.max_cut_velocity
                dir_ok = self.cut_direction_sign is None or np.sign(p1 - p0) == self.cut_direction_sign
                if self.max_other_axis_range is not None:
                    mask = (t >= t0) & (t <= t1)
                    other_ok = all(
                        float(op[mask].max() - op[mask].min()) <= self.max_other_axis_range
                        for op in other_axes_pos.values()
                    ) if mask.any() else True
                else:
                    other_ok = True
                if vel_ok and dir_ok and other_ok:
                    kept.append((t0, t1))
                else:
                    n_filtered += 1
            filters_desc = ", ".join(
                s for s in (
                    f"max_cut_velocity={self.max_cut_velocity}" if self.max_cut_velocity is not None else None,
                    f"cut_direction_sign={self.cut_direction_sign}" if self.cut_direction_sign is not None else None,
                    f"max_other_axis_range={self.max_other_axis_range}" if self.max_other_axis_range is not None else None,
                ) if s is not None
            )
            if kept:
                msgs.append(
                    f"Segmentfilter ({filters_desc}): {n_filtered} von {len(cuts)} "
                    f"Rohsegmenten als Ruecklauf/Leerfahrt verworfen (dominante Achse "
                    f"axis{dom_axis}), {len(kept)} als Schnitte behalten."
                )
                cuts = kept
            else:
                msgs.append(
                    f"Segmentfilter ({filters_desc}) haette alle {len(cuts)} Rohsegmente "
                    f"verworfen -- Filter fuer diesen Block ignoriert, unfiltrierte "
                    f"Segmente werden zurueckgegeben."
                )

        if dom_axis is not None and len(cuts) != self.cuts_per_block:
            msgs.append(
                f"Achtung: {len(cuts)} Schnitte erkannt (dominante Achse "
                f"axis{dom_axis}, Spannweite {dom_axis_range:.2f}), "
                f"erwartet wurden {self.cuts_per_block}. Bitte "
                f"reversal_prominence_frac / reversal_smooth_window pruefen "
                f"oder Bloecke manuell inspizieren."
            )
        return cuts, msgs

    def cut_diagnostics(self, block: dict) -> pd.DataFrame:
        """
        Diagnose-Tabelle ueber ALLE rohen Schnitt-Segmente eines Blocks (vor
        max_cut_velocity-/cut_direction_sign-/max_other_axis_range-
        Filterung): Dauer, Bewegungsrichtung und mittlere Geschwindigkeit
        der dominanten Vorschubachse, maximale Positions-Spannweite der
        NICHT gewaehlten Vorschubachsen (other_axis_max_range), sowie
        Kraftamplitude (nur zur Information/Kreuzpruefung, NICHT Teil der
        Filterentscheidung) je Segment. direction_sign/mean_velocity/
        other_axis_max_range/kept_with_*_filter sind NaN, wenn kein
        verlaessliches Achssignal vorliegt (Fallback-Faelle) bzw. wenn der
        jeweilige Filter nicht gesetzt ist. kept_combined ist die UND-
        Verknuepfung aller (nur der tatsaechlich aktiven) Filter.

        Gedacht zum Vor-Pruefen im Notebook (Abschnitt 3), BEVOR
        max_cut_velocity/cut_direction_sign/max_other_axis_range scharf
        geschaltet werden -- analog zu burst_diagnostics() fuer
        min_burst_amplitude.
        """
        cuts, _msgs, dom_axis, _range, t, pos, other_axes_pos = self._cut_segments_raw(block)
        rows = []
        for i, (t0, t1) in enumerate(cuts, start=1):
            dur = t1 - t0
            amp = self._burst_amplitude(t0, t1)
            if dom_axis is not None and dur > 0:
                p0 = float(np.interp(t0, t, pos))
                p1 = float(np.interp(t1, t, pos))
                direction_sign = float(np.sign(p1 - p0))
                mean_velocity = abs(p1 - p0) / dur
                mask = (t >= t0) & (t <= t1)
                other_axis_max_range = (
                    max(float(op[mask].max() - op[mask].min()) for op in other_axes_pos.values())
                    if mask.any() and other_axes_pos else np.nan
                )
                kept_with_velocity_filter = (
                    bool(mean_velocity <= self.max_cut_velocity)
                    if self.max_cut_velocity is not None else np.nan
                )
                kept_with_direction_filter = (
                    bool(direction_sign == self.cut_direction_sign)
                    if self.cut_direction_sign is not None else np.nan
                )
                kept_with_other_axis_filter = (
                    bool(other_axis_max_range <= self.max_other_axis_range)
                    if self.max_other_axis_range is not None and not np.isnan(other_axis_max_range)
                    else np.nan
                )
                active = [
                    v for v in (
                        kept_with_velocity_filter, kept_with_direction_filter,
                        kept_with_other_axis_filter,
                    )
                    if not (isinstance(v, float) and np.isnan(v))
                ]
                kept_combined = bool(all(active)) if active else np.nan
            else:
                direction_sign = np.nan
                mean_velocity = np.nan
                other_axis_max_range = np.nan
                kept_with_velocity_filter = np.nan
                kept_with_direction_filter = np.nan
                kept_with_other_axis_filter = np.nan
                kept_combined = np.nan
            rows.append({
                "cut_id": i, "t_start": t0, "t_end": t1, "duration_s": dur,
                "direction_sign": direction_sign, "mean_velocity": mean_velocity,
                "other_axis_max_range": other_axis_max_range,
                "force_amplitude": amp,
                "kept_no_filter": True,
                "kept_with_velocity_filter": kept_with_velocity_filter,
                "kept_with_direction_filter": kept_with_direction_filter,
                "kept_with_other_axis_filter": kept_with_other_axis_filter,
                "kept_combined": kept_combined,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 6. Einen Schnitt extrahieren: Rohsignal-Tabelle + Winkel/Radialkraft
    # ------------------------------------------------------------------
    def extract_cut(
        self, block_id: int, cut_id: int, t_start: float, t_end: float
    ) -> CutResult:
        warn_msgs: List[str] = []

        # --- Kraftkanaele (20 kHz, ADC-Rohwerte -> Newton) ---
        force_raw = {}
        force_n = {}
        t_force_ref = None
        for si in self.FORCE_SENSOR_INDICES:
            t, raw = self._concat_force_channel(si, t_start, t_end)
            if t.size > 0:
                mask = (t >= t_start) & (t <= t_end)
                t, raw = t[mask], raw[mask]
            if t.size == 0:
                warn_msgs.append(f"force_{si}: keine Daten im Schnittfenster.")
                continue
            force_raw[f"force_{si}"] = (t, raw)
            force_n[f"force_{si}"] = MSU.adc_counts_to_force(
                raw, bits=self.adc_bits, f_range=self.adc_f_range, bipolar=self.adc_bipolar
            )
            if t_force_ref is None:
                t_force_ref = t

        # --- Achsdaten (250 Hz: Spindel + Vorschubachsen) ---
        mdf = self._machine_frame(t_start, t_end)
        if not mdf.empty:
            mdf = mdf[(mdf["t"] >= t_start) & (mdf["t"] <= t_end)].reset_index(drop=True)
        if mdf.empty:
            warn_msgs.append("Keine 'machine'-Achsdaten im Schnittfenster (nur Kraftkanaele exportiert).")

        # --- SinuTrace-Stil-Rohtabelle: outer join aller Kanaele auf Zeit ---
        # Kraftkanaele koennen -- je nach Burst-Dokumenten im JSON --
        # innerhalb desselben Fensters leicht unterschiedliche Sample-Anzahlen
        # haben; daher per Zeitstempel (nicht positionsbasiert) zusammenfuehren.
        raw_table = None
        force_sample_period = 1.0 / 20000.0
        for name, (t, _) in force_raw.items():
            chan_df = pd.DataFrame({"time": t, name: force_n[name]}).sort_values("time")
            if raw_table is None:
                raw_table = chan_df
            else:
                raw_table = pd.merge_asof(
                    raw_table,
                    chan_df,
                    on="time",
                    direction="nearest",
                    tolerance=force_sample_period / 2,
                )
        if not mdf.empty:
            axis_cols = [c for c in mdf.columns if c.startswith("axis")]
            mach_part = mdf[["t"] + axis_cols].rename(columns={"t": "time"})
            if raw_table is None:
                raw_table = mach_part
            else:
                raw_table = pd.merge_asof(
                    raw_table.sort_values("time"),
                    mach_part.sort_values("time"),
                    on="time",
                    direction="nearest",
                    tolerance=0.004,
                )
        if raw_table is None:
            raw_table = pd.DataFrame(columns=["time"])
        raw_table = raw_table.sort_values("time").reset_index(drop=True)

        # --- Winkel hochinterpolieren (Spindel, 0-360 deg) ---
        angle_deg = np.array([])
        if not mdf.empty and f"axis{self.spindle_axis_index}_positionact" in mdf.columns:
            enc = mdf[f"axis{self.spindle_axis_index}_positionact"].values
            t_enc = mdf["t"].values
            t_grid_angle = np.linspace(t_start, t_end, self.n_angle_samples)
            angle_deg = MSU.phi_from_encoder_deg(enc, t_enc, t_grid_angle)
        else:
            warn_msgs.append("Keine Spindelposition verfuegbar -- Winkel-Array leer.")

        # --- Kraft herunterinterpolieren (n_force_samples, geglaettet) ---
        force_ds: Dict[str, np.ndarray] = {}
        if t_force_ref is not None:
            for name, v in force_n.items():
                t_ch, _ = force_raw[name]
                v_smooth = MSU.maf(v, n=max(1, int(len(v) / self.n_force_samples)))
                _, v_ds = MSU.resample_linear_to_n(
                    v_smooth, t_ch, self.n_force_samples, t_start=t_start, t_end=t_end
                )
                force_ds[name] = v_ds

        # --- Radialkraft an der Schneidkante + Winkel-Summenprofil ---
        radial_profile = pd.DataFrame(columns=["angle_deg", "Fr_sum", "Fr_mean", "count"])
        if angle_deg.size and "force_0" in force_ds and "force_1" in force_ds:
            phi_rad = np.deg2rad(angle_deg)
            Fr = MSU.radial_force(force_ds["force_0"], force_ds["force_1"], phi_rad)
            radial_profile = MSU.sum_radial_force_by_angle(Fr, angle_deg, n_bins=360, degrees=True)
        else:
            warn_msgs.append(
                "Radialkraft-Profil nicht berechnet (fehlender Winkel oder fehlende Fx/Fy)."
            )

        return CutResult(
            block_id=block_id,
            cut_id=cut_id,
            t_start=t_start,
            t_end=t_end,
            raw_table=raw_table,
            angle_deg=angle_deg,
            force_ds=force_ds,
            radial_force_profile=radial_profile,
            warnings=warn_msgs,
        )

    # ------------------------------------------------------------------
    # 7. Kompletten Lauf orchestrieren
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """
        Fuehrt die komplette Pipeline aus: load() muss vorher aufgerufen
        worden sein. Schreibt je Schnitt ein CSV nach output_dir und
        liefert eine Zusammenfassungs-Tabelle (auch als self.summary_
        verfuegbar).
        """
        if not self._force_docs[0]:
            self.load()

        blocks = self.detect_blocks()
        summary_rows = []
        self.results = []

        for block_id, block in enumerate(blocks, start=1):
            cuts, block_msgs = self.detect_cuts_in_block(block)
            for msg in block_msgs:
                warnings.warn(f"[Block {block_id}] {msg}")

            for cut_id, (t0, t1) in enumerate(cuts, start=1):
                result = self.extract_cut(block_id, cut_id, t0, t1)
                fname = f"Schnitt_block{block_id:02d}_cut{cut_id:02d}.csv"
                fpath = os.path.join(self.output_dir, fname)
                result.raw_table.to_csv(fpath, index=False)
                result.csv_path = fpath

                profile_fname = f"Schnitt_block{block_id:02d}_cut{cut_id:02d}_radialforce_by_angle.csv"
                profile_fpath = os.path.join(self.output_dir, profile_fname)
                result.radial_force_profile.to_csv(profile_fpath, index=False)

                for msg in result.warnings:
                    warnings.warn(f"[Block {block_id} / Schnitt {cut_id}] {msg}")

                self.results.append(result)
                summary_rows.append(
                    {
                        "block_id": block_id,
                        "cut_id": cut_id,
                        "t_start_s": t0,
                        "t_end_s": t1,
                        "duration_s": t1 - t0,
                        "n_rows_raw": len(result.raw_table),
                        "csv_path": fpath,
                        "n_warnings": len(result.warnings),
                    }
                )

        self.summary_ = pd.DataFrame(summary_rows)
        print(
            f"[run] {len(blocks)} Bloecke, {len(self.results)} Schnitte extrahiert. "
            f"CSVs liegen in: {self.output_dir}"
        )
        return self.summary_

    # ------------------------------------------------------------------
    # 8. Alternative Segmentierung: luecken-freie Aufnahme-Fenster
    #    (Recording ist laut NC-Programm nur waehrend eines Schnitts aktiv)
    # ------------------------------------------------------------------
    def gap_free_segments(self) -> List[dict]:
        """
        Oeffentlicher Zugriff auf _force_bursts(): alle zusammenhaengenden
        (luecken-freien) Aufnahme-Fenster, UNGEFILTERT (auch kurze
        "Heartbeat"-Bursts, anders als detect_blocks()). Laut Analyse des
        NC-Programms ist die Aufzeichnung nur waehrend eines Schnitts aktiv
        und wird danach abgeschaltet -- jeder Burst entspricht also genau
        einer echten Aufzeichnungssitzung, unabhaengig von der
        Richtungsumkehr-basierten Schnitt-Segmentierung
        (detect_cuts_in_block()/run()).
        """
        return self._force_bursts()

    def export_gap_free_segments(self, output_dir: str = "./schnitte_gap_free") -> pd.DataFrame:
        """
        Exportiert JEDES luecken-freie Aufnahme-Fenster (gap_free_segments())
        als eigene CSV-Datei nach output_dir -- unabhaengig von
        detect_blocks()/detect_cuts_in_block()/run() und OHNE Dauer-/
        Amplituden-Filterung (jeder Burst wird gespeichert, auch kurze). Je
        CSV identisches Format zu extract_cut().raw_table (SinuTrace-Stil:
        'time', 'force_0/1/2' in Volt, Achsdaten), aber ohne die Winkel-/
        Radialkraft-Zusatzberechnung (nicht Teil dieses Exports).
        """
        os.makedirs(output_dir, exist_ok=True)
        segments = self.gap_free_segments()
        summary_rows = []
        for seg_id, b in enumerate(segments, start=1):
            t0, t1 = b["t_start"], b["t_end"]
            result = self.extract_cut(block_id=0, cut_id=seg_id, t_start=t0, t_end=t1)
            fname = f"gap_free_{seg_id:03d}_t{t0:.2f}-{t1:.2f}.csv"
            fpath = os.path.join(output_dir, fname)
            result.raw_table.to_csv(fpath, index=False)
            summary_rows.append({
                "segment_id": seg_id, "t_start": t0, "t_end": t1,
                "duration_s": t1 - t0, "n_rows_raw": len(result.raw_table),
                "csv_path": fpath, "n_warnings": len(result.warnings),
            })
        summary = pd.DataFrame(summary_rows)
        print(
            f"[export_gap_free_segments] {len(segments)} luecken-freie Segmente exportiert. "
            f"CSVs liegen in: {output_dir}"
        )
        return summary

    # ------------------------------------------------------------------
    # 9. Alternative Gruppierung: Bloecke nach Z-Tiefe (axis6) zu
    #    "Leveln" zusammenfassen (Standzeit-Versuch: ap konstant fuer
    #    8 Schnitte = 2 Bloecke, Drehzahl/Vorschub konstant fuer 4
    #    Schnitte = 1 Block)
    # ------------------------------------------------------------------
    def z_level_diagnostics(
        self,
        z_axis_index: Optional[int] = None,
        z_tol: float = 0.05,
        z_safe_heights: Sequence[float] = (10.0, 20.0),
    ) -> pd.DataFrame:
        """
        Gruppiert die Bloecke aus detect_blocks() nach Z-Tiefe (per Default
        axis6 = letzte Vorschubachse in feed_axis_indices) statt nach Zeit.
        Hintergrund (Standzeit-Versuch, NUT_CUTO_DA_Y.MPF): die axiale
        Schnitttiefe (R104/ap) wird zwischen Versuchen von Hand geaendert und
        bleibt fuer 8 Schnitte (= 2 Bloecke) konstant; Drehzahl/Vorschub
        aendern sich alle 4 Schnitte (= 1 Block). Z=10mm ist die
        Eilgang-Sicherheitshoehe ZWISCHEN Schnitten innerhalb eines Levels
        (kurzes Transienten-Segment innerhalb eines Blocks -- deshalb Median
        statt Mittelwert je Block: >90% der Samples liegen auf der echten
        Schnitttiefe, der Eilgang-Anteil ist klein aber verzerrt den
        Mittelwert/die Standardabweichung stark). Z=20mm ist die
        Sicherheitshoehe NACH allen 8 Schnitten eines Levels (Level-Ende) --
        Bloecke, deren Z-Median dort liegt, sind daher keine echten
        Schnitte, sondern Leerlauf-Aufzeichnungen an der Park-Position.

        Parameters
        ----------
        z_axis_index : int, optional
            Achsindex der Z-Achse. Default: feed_axis_indices[-1] (axis6 in
            diesem Notebook-Setup).
        z_tol : float, default 0.05
            Toleranz (mm) fuer "gleiche Z-Tiefe" -- sowohl fuer die
            Level-Gruppierung aufeinanderfolgender Bloecke als auch (skaliert)
            fuer frac_near_median und den Vergleich mit z_safe_heights.
        z_safe_heights : Sequence[float], default (10.0, 20.0)
            Bekannte Sicherheits-/Park-Hoehen (siehe oben) -- Bloecke mit
            Z-Median nahe einer dieser Hoehen werden als near_safe_height=True
            markiert (voraussichtlich kein echter Schnitt).

        Returns
        -------
        Eine Zeile je Block: block_id, level_id, t_start, t_end, duration_s,
        z_median, z_mad, frac_near_median, near_safe_height,
        n_blocks_in_level. Keine Filterung -- die Spalten zeigen lediglich
        Auffaelligkeiten an (analog zu burst_diagnostics()/cut_diagnostics()).
        """
        if z_axis_index is None:
            z_axis_index = self.feed_axis_indices[-1]
        col = f"axis{z_axis_index}_positionact"

        blocks = self.detect_blocks()
        rows = []
        for block_id, block in enumerate(blocks, start=1):
            t0, t1 = block["t_start"], block["t_end"]
            mdf = self._machine_frame(t0, t1)
            if not mdf.empty:
                mdf = mdf[(mdf["t"] >= t0) & (mdf["t"] <= t1)]
            if mdf.empty or col not in mdf.columns:
                rows.append({
                    "block_id": block_id, "t_start": t0, "t_end": t1,
                    "duration_s": t1 - t0, "z_median": np.nan, "z_mad": np.nan,
                    "frac_near_median": np.nan, "near_safe_height": False,
                })
                continue
            z = mdf[col].to_numpy()
            z_median = float(np.median(z))
            z_mad = float(np.median(np.abs(z - z_median)))
            frac_near_median = float(np.mean(np.abs(z - z_median) < z_tol * 20))
            near_safe_height = bool(np.any(np.abs(z_median - np.asarray(z_safe_heights)) <= z_tol))
            rows.append({
                "block_id": block_id, "t_start": t0, "t_end": t1,
                "duration_s": t1 - t0, "z_median": z_median, "z_mad": z_mad,
                "frac_near_median": frac_near_median, "near_safe_height": near_safe_height,
            })

        level_id = 0
        prev_z = None
        level_ids = []
        for row in rows:
            z = row["z_median"]
            if prev_z is None or np.isnan(z) or np.isnan(prev_z) or abs(z - prev_z) > z_tol:
                level_id += 1
            level_ids.append(level_id)
            prev_z = z
        for row, lid in zip(rows, level_ids):
            row["level_id"] = lid

        diag = pd.DataFrame(rows)
        level_counts = diag.groupby("level_id")["block_id"].transform("count")
        diag["n_blocks_in_level"] = level_counts
        return diag[[
            "block_id", "level_id", "t_start", "t_end", "duration_s",
            "z_median", "z_mad", "frac_near_median", "near_safe_height",
            "n_blocks_in_level",
        ]]

    def export_z_level_segments(
        self,
        output_dir: str = "./schnitte_z_level",
        z_axis_index: Optional[int] = None,
        z_tol: float = 0.05,
    ) -> pd.DataFrame:
        """
        Exportiert je Z-Tiefen-Level (siehe z_level_diagnostics()) EINE CSV,
        die die Rohsignale aller Bloecke dieses Levels (alle Schnitte
        derselben Schnitttiefe) aneinandergehaengt enthaelt -- Format
        identisch zu extract_cut().raw_table (SinuTrace-Stil). Bloecke mit
        z_median > 0 werden verworfen (Eilgang-/Park-Hoehen wie Z=10/20mm,
        siehe z_level_diagnostics() -- kein echter Schnitt, da die
        Schnitttiefe von der Z=0-Oberflaeche aus negativ ist). Bloecke mit
        NaN-z_median (fehlende Achsdaten) bleiben wie bisher erhalten. Vor dem
        Export werden vorhandene `level_*.csv`-Dateien im output_dir geloescht,
        damit Level, die durch die z_median > 0 - Filterung wegfallen, nicht
        als Datei-Leiche eines frueheren Laufs erhalten bleiben.
        """
        os.makedirs(output_dir, exist_ok=True)
        for stale in Path(output_dir).glob("level_*.csv"):
            stale.unlink()
        z_diag = self.z_level_diagnostics(z_axis_index=z_axis_index, z_tol=z_tol)
        z_diag = z_diag[~(z_diag["z_median"] > 0)]
        blocks = self.detect_blocks()

        summary_rows = []
        for level_id, group in z_diag.groupby("level_id"):
            tables = []
            for _, row in group.iterrows():
                block = blocks[int(row["block_id"]) - 1]
                result = self.extract_cut(
                    block_id=int(row["block_id"]), cut_id=0,
                    t_start=block["t_start"], t_end=block["t_end"],
                )
                tables.append(result.raw_table)
            merged = pd.concat(tables, ignore_index=True).sort_values("time").reset_index(drop=True)
            z_depth = float(group["z_median"].median())
            fname = f"level_{int(level_id):02d}_z{z_depth:.2f}.csv"
            fpath = os.path.join(output_dir, fname)
            merged.to_csv(fpath, index=False)
            summary_rows.append({
                "level_id": int(level_id), "z_depth": z_depth,
                "n_blocks": len(group), "block_ids": list(group["block_id"]),
                "near_safe_height": bool(group["near_safe_height"].any()),
                "n_rows_raw": len(merged), "csv_path": fpath,
            })

        summary = pd.DataFrame(summary_rows)
        print(
            f"[export_z_level_segments] {len(summary)} Z-Level exportiert. "
            f"CSVs liegen in: {output_dir}"
        )
        return summary
