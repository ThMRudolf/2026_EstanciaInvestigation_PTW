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
     werden durch Richtungsumkehr der dominanten Vorschubachse getrennt
     (siehe detect_cuts_in_block).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Sequence, Tuple

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
        if "$date" in value:
            return value["$date"]
        if "$oid" in value:
            return value["$oid"]
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


@dataclass
class CutResult:
    """Container fuer einen einzelnen extrahierten Schnitt."""

    block_id: int
    cut_id: int
    t_start: float
    t_end: float
    raw_table: pd.DataFrame          # SinuTrace-Stil, Rohsignale
    angle_deg: np.ndarray            # (n_angle_samples,) hochinterpolierte Spindelposition
    force_ds: Dict[str, np.ndarray]  # herunterinterpolierte Kraftkanaele (Volt)
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
        Aufloesung des Kraft-ADC (siehe MSU.adc_counts_to_voltage).
    adc_v_range : float, default 10.0
        Spannungsbereich des Kraft-ADC in Volt (siehe
        MSU.adc_counts_to_voltage).
    adc_bipolar : bool, default True
        Ob der ADC bipolar (+/-10 V) oder unipolar (0-10 V) ist.
    min_block_duration_s : float, default 3.0
        Minimale Dauer eines zusammenhaengenden Force-Bursts, damit er als
        "Bearbeitungsblock" (und nicht als Leerlauf-Heartbeat-Sample)
        gewertet wird.
    reversal_smooth_window : int, default 15
        Fensterbreite (Anzahl 250-Hz-Samples) der Glaettung vor der
        Richtungsumkehr-Erkennung der Vorschubachse.
    reversal_prominence_frac : float, default 0.1
        Mindest-Prominenz eines Richtungsumkehr-Peaks, als Anteil der
        Positions-Spannweite der dominanten Vorschubachse im Block.
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
        cuts_per_block: int = 4,
        n_angle_samples: int = 1000,
        n_force_samples: int = 1000,
        adc_bits: int = 16,
        adc_v_range: float = 10.0,
        adc_bipolar: bool = True,
        min_block_duration_s: float = 3.0,
        reversal_smooth_window: int = 15,
        reversal_prominence_frac: float = 0.1,
        min_feed_range: float = 2.0,
        r_tool_mm: Optional[float] = None,
    ):
        self.json_path = json_path
        self.output_dir = output_dir
        self.spindle_axis_index = spindle_axis_index
        self.feed_axis_indices = tuple(feed_axis_indices)
        self.cuts_per_block = cuts_per_block
        self.n_angle_samples = n_angle_samples
        self.n_force_samples = n_force_samples
        self.adc_bits = adc_bits
        self.adc_v_range = adc_v_range
        self.adc_bipolar = adc_bipolar
        self.min_block_duration_s = min_block_duration_s
        self.reversal_smooth_window = reversal_smooth_window
        self.reversal_prominence_frac = reversal_prominence_frac
        self.min_feed_range = min_feed_range
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
        return df

    def machine_frame(self, t_start: float, t_end: float) -> pd.DataFrame:
        """Oeffentlicher Zugriff auf _machine_frame() fuer Diagnose-Plots im Notebook."""
        return self._machine_frame(t_start, t_end)

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
        """
        bursts = self._force_bursts()
        blocks = [b for b in bursts if (b["t_end"] - b["t_start"]) >= self.min_block_duration_s]
        print(
            f"[detect_blocks] {len(bursts)} Force-Bursts gefunden, "
            f"{len(blocks)} davon >= {self.min_block_duration_s}s "
            f"(als Bearbeitungsbloecke gewertet)."
        )
        return blocks

    # ------------------------------------------------------------------
    # 5. Einzelne Schnitte innerhalb eines Blocks trennen
    #    (Vorschubachsen-Richtungsumkehr)
    # ------------------------------------------------------------------
    def detect_cuts_in_block(self, block: dict) -> Tuple[List[Tuple[float, float]], List[str]]:
        """
        Erkennt die Grenzen der einzelnen Schnitte innerhalb eines Blocks
        anhand der Richtungsumkehr der Positions-Signale der konfigurierten
        Vorschubachsen (feed_axis_indices). Die Achse mit der groessten
        Positions-Spannweite im Block wird als "dominante" Vorschubachse
        gewaehlt.

        Fallback: Wenn keine 'machine'-Daten das Blockfenster ueberlappen
        oder keine Richtungsumkehr erkannt wird, wird der Block in
        cuts_per_block gleich lange Zeitabschnitte geteilt (mit Warnung).

        Returns
        -------
        cuts : list of (t_start, t_end)
        msgs : list of Warnhinweise (fuer Protokollierung)
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
            return list(zip(edges[:-1], edges[1:])), msgs

        # dominante Vorschubachse = groesste Positions-Spannweite
        ranges = {
            ai: mdf[f"axis{ai}_positionact"].max() - mdf[f"axis{ai}_positionact"].min()
            for ai in self.feed_axis_indices
        }
        dom_axis = max(ranges, key=ranges.get)
        pos = mdf[f"axis{dom_axis}_positionact"].values
        t = mdf["t"].values

        if ranges[dom_axis] < self.min_feed_range:
            msgs.append(
                f"Keine Vorschubachse bewegt sich um mehr als min_feed_range="
                f"{self.min_feed_range} (groesste Spannweite: axis{dom_axis} = "
                f"{ranges[dom_axis]:.3f}) -- vermutlich reiner Spindel-Leerlauf "
                f"oder zu kurzes/verrauschtes Machine-Datenfenster. "
                f"Fallback: gleichmaessige Teilung in {self.cuts_per_block} Schnitte."
            )
            edges = np.linspace(t_start, t_end, self.cuts_per_block + 1)
            return list(zip(edges[:-1], edges[1:])), msgs

        pos_smooth = (
            pd.Series(pos)
            .rolling(self.reversal_smooth_window, center=True, min_periods=1)
            .mean()
            .values
        )
        prom = ranges[dom_axis] * self.reversal_prominence_frac
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

        if len(cuts) != self.cuts_per_block:
            msgs.append(
                f"Achtung: {len(cuts)} Schnitte erkannt (dominante Achse "
                f"axis{dom_axis}, Spannweite {ranges[dom_axis]:.2f}), "
                f"erwartet wurden {self.cuts_per_block}. Bitte "
                f"reversal_prominence_frac / reversal_smooth_window pruefen "
                f"oder Bloecke manuell inspizieren."
            )
        return cuts, msgs

    # ------------------------------------------------------------------
    # 6. Einen Schnitt extrahieren: Rohsignal-Tabelle + Winkel/Radialkraft
    # ------------------------------------------------------------------
    def extract_cut(
        self, block_id: int, cut_id: int, t_start: float, t_end: float
    ) -> CutResult:
        warn_msgs: List[str] = []

        # --- Kraftkanaele (20 kHz, ADC-Rohwerte -> Volt) ---
        force_raw = {}
        force_v = {}
        t_force_ref = None
        for si in self.FORCE_SENSOR_INDICES:
            t, raw = self._concat_force_channel(si, t_start, t_end)
            if t.size == 0:
                warn_msgs.append(f"force_{si}: keine Daten im Schnittfenster.")
                continue
            mask = (t >= t_start) & (t <= t_end)
            t, raw = t[mask], raw[mask]
            force_raw[f"force_{si}"] = (t, raw)
            force_v[f"force_{si}"] = MSU.adc_counts_to_voltage(
                raw, bits=self.adc_bits, v_range=self.adc_v_range, bipolar=self.adc_bipolar
            )
            if t_force_ref is None:
                t_force_ref = t

        # --- Achsdaten (250 Hz: Spindel + Vorschubachsen) ---
        mdf = self._machine_frame(t_start, t_end)
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
            chan_df = pd.DataFrame({"time": t, name: force_v[name]}).sort_values("time")
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
            for name, v in force_v.items():
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
