"""
laufnummer_correction.py
-------------------------
Preprocessing-Schritt: korrigiert die Laufnummer (workpiecedata[0].serialnr[3])
in axon_logger_ptw_LevNNN.raw.json-Exporten.

Hintergrund: Jede LevNNN-Datei ist ein kumulativer Dump (enthaelt alle bisher
in der Session aufgezeichneten Dokumente, nicht nur die des aktuellen Laufs).
Die Laufnummer wird vom Bediener von Hand am Maschinen-Panel hochgezaehlt und
ist daher fehleranfaellig: in der Praxis wurden mehrere Fehlerbilder beobachtet
(20260701_C45_y: ein Lauf wurde vergessen hochzuzaehlen, zwei echte Laeufe
landeten unter derselben Nummer und ohne jedes Rohdaten-Signal dazwischen;
20260630_ALU_y: die allererste Datei enthaelt bereits 7 verschiedene
Laufnummern, und eine bereits verbrauchte Nummer taucht spaeter erneut auf;
20260701_ALU_x/20260701_C45_y: Laufnummern werden teils uebersprungen).

Korrekturstrategie: Die Dokument-Identitaet (_id.$oid, MongoDB-ObjectId) zeigt
exakt, welche Dokumente seit dem letzten Dump neu hinzugekommen sind (Dateien
sind kumulativ). Innerhalb der neu hinzugekommenen Dokumente einer Datei
werden Aenderungen des rohen Laufnummer-Tags in zeitlicher Reihenfolge
verfolgt: jede Aenderung des rohen Tag-Werts (in beliebige Richtung, auch bei
Wiederverwendung einer alten Nummer) wird als echter Laufwechsel gewertet und
erhoeht den korrigierten Zaehler um 1 -- unabhaengig vom konkreten rohen Wert.
Das deckt sowohl "Datei enthaelt mehrere Laeufe" als auch "Nummer wurde
wiederverwendet/uebersprungen" korrekt ab, rein aus der Reihenfolge der
Tag-Wechsel. Faelle OHNE jedes Tag-Wechsel-Signal (z.B. der C45_y-Fall, wo
zwei Laeufe identisch getaggt UND ohne Dateigrenze dazwischen aufgezeichnet
wurden) koennen dadurch nicht automatisch getrennt werden -- solche Gruppen
bleiben bewusst zusammengefasst und werden in der Zusammenfassung als
auffaellig gross markiert (siehe `laufnummer_group_summary`), zur manuellen
Pruefung.

Vertraulichkeitshinweis: siehe cnc_cut_extractor.py -- dieselben Rohdaten.
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple, Union

import ijson
import pandas as pd

PathLike = Union[str, Path]

_LEV_RE = re.compile(r"[Ll]ev(\d+)")


def _lev_number(path: Path) -> int:
    """Extrahiert die LevNNN-Nummer aus einem Dateinamen wie 'axon_logger_ptw_Lev020.raw.json'
    oder 'axon_logger_ptw_lev002.raw.json' (Gross-/Kleinschreibung uneinheitlich)."""
    m = _LEV_RE.search(path.name)
    if not m:
        raise ValueError(f"Kann keine Lev-Nummer aus Dateiname ableiten: {path.name}")
    return int(m.group(1))


def list_lev_files(input_dir: PathLike) -> List[Path]:
    """Alle axon_logger_ptw_Lev*.raw.json-Dateien in input_dir, aufsteigend nach LevNNN sortiert."""
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("axon_logger_ptw_*.raw.json"), key=_lev_number)
    if not files:
        raise ValueError(f"Keine axon_logger_ptw_Lev*.raw.json-Dateien in {input_dir} gefunden.")
    return files


def resolve_input_output(session_dir: PathLike) -> Tuple[Path, Path]:
    """
    Bestimmt Input-/Output-Verzeichnis fuer eine Session:
    - falls session_dir/raw existiert, wird dies als Input verwendet (Konvention:
      raw = unveraenderte Rohdaten, prep = hier erzeugte korrigierte Kopien);
    - sonst wird session_dir selbst als Input verwendet.
    Output ist in beiden Faellen session_dir/prep (wird angelegt, falls noetig).
    """
    session_dir = Path(session_dir)
    raw_dir = session_dir / "raw"
    input_dir = raw_dir if raw_dir.is_dir() else session_dir
    output_dir = session_dir / "prep"
    return input_dir, output_dir


def _doc_id(doc: dict) -> str:
    oid = doc.get("_id")
    if isinstance(oid, dict) and "$oid" in oid:
        return oid["$oid"]
    return str(oid)


def _doc_time(doc: dict) -> Optional[str]:
    t = doc.get("time")
    if isinstance(t, dict) and "$date" in t:
        return t["$date"]
    return None


def _get_laufnummer(doc: dict) -> Optional[int]:
    """Liest workpiecedata[0].serialnr[3] (Laufnummer), None falls Feld fehlt/unerwartet geformt."""
    wp = doc.get("workpiecedata")
    if not wp:
        return None
    serialnr = wp[0].get("serialnr")
    if not serialnr or len(serialnr) < 4:
        return None
    val = serialnr[3]
    if isinstance(val, dict) and "$numberLong" in val:
        return int(val["$numberLong"])
    if isinstance(val, (int, float)):
        return int(val)
    return None


def _set_laufnummer(doc: dict, new_value: int) -> None:
    """Setzt workpiecedata[0].serialnr[3] auf new_value, alle anderen Felder bleiben unveraendert.
    Betrifft NUR das dokument-weite workpiecedata-Feld, nicht das (unabhaengige, hier nicht
    beruehrte) verschachtelte 'serialnr'-Feld innerhalb einzelner machine-Snapshots ('data')."""
    wp = doc.get("workpiecedata")
    if not wp:
        return
    serialnr = wp[0].get("serialnr")
    if not serialnr or len(serialnr) < 4:
        return
    val = serialnr[3]
    if isinstance(val, dict) and "$numberLong" in val:
        val["$numberLong"] = str(new_value)
    else:
        serialnr[3] = new_value


def correct_session_folder(input_dir: PathLike, output_dir: PathLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fuehrt die Laufnummer-Korrektur ueber alle axon_logger_ptw_Lev*.raw.json-Dateien in
    input_dir aus (aufsteigend nach LevNNN) und schreibt je Datei eine korrigierte Kopie
    (gleicher Dateiname) nach output_dir.

    Je Datei zwei Durchlaeufe: (1) leichtgewichtiger Scan, der fuer neu hinzugekommene
    Dokumente (_id noch nicht bekannt) nur _id/time/rohe-Laufnummer sammelt, diese nach
    Zeit sortiert und Tag-Wechsel erkennt (jeder Wechsel = neuer korrigierter Zaehlerstand,
    unabhaengig vom rohen Wert -- deckt sowohl "mehrere Laeufe in einer Datei" als auch
    "Nummer wiederverwendet/uebersprungen" ab); (2) Schreib-Durchlauf, der die so ermittelte
    Zuordnung auf alle Dokumente (neue + uebernommene) anwendet.

    Gibt zwei Tabellen zurueck:
    - file_summary: eine Zeile je Datei (n_docs, n_new_docs, n_tag_wechsel, ...).
    - group_summary: eine Zeile je korrigierter Laufnummer (n_docs), mit einem
      `anomalous`-Flag fuer Gruppen, die deutlich groesser sind als der Median (moegliches
      Anzeichen fuer einen nicht automatisch trennbaren Merge wie im C45_y-Fall).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_lev_files(input_dir)

    seen_ids: Dict[str, int] = {}
    last_raw_seen: Optional[int] = None
    corrected_counter: Optional[int] = None
    file_summary_rows = []

    for i, path in enumerate(files):
        # --- Durchlauf 1: leichtgewichtiger Scan neuer Dokumente ---
        new_meta: List[Tuple[str, Optional[str], Optional[int]]] = []
        n_docs = 0
        with open(path, "rb") as f:
            for doc in ijson.items(f, "item", use_float=True):
                n_docs += 1
                doc_id = _doc_id(doc)
                if doc_id in seen_ids:
                    continue
                new_meta.append((doc_id, _doc_time(doc), _get_laufnummer(doc)))

        new_meta.sort(key=lambda x: (x[1] is None, x[1]))

        n_transitions = 0
        n_missing_tag = 0
        for doc_id, _t, raw_val in new_meta:
            if raw_val is None:
                n_missing_tag += 1
            if corrected_counter is None:
                corrected_counter = raw_val if raw_val is not None else 0
            elif raw_val is not None and raw_val != last_raw_seen:
                corrected_counter += 1
                n_transitions += 1
            seen_ids[doc_id] = corrected_counter
            if raw_val is not None:
                last_raw_seen = raw_val

        if n_missing_tag:
            warnings.warn(
                f"{path.name}: {n_missing_tag} neue Dokumente ohne lesbare Laufnummer "
                f"(workpiecedata/serialnr fehlt oder unerwartet geformt) -- diese erben den "
                f"zuletzt gueltigen korrigierten Wert."
            )

        # --- Durchlauf 2: Schreiben der korrigierten Kopie ---
        out_path = output_dir / path.name
        with open(path, "rb") as fin, open(out_path, "w", encoding="utf-8") as fout:
            fout.write("[")
            first_written = True
            for doc in ijson.items(fin, "item", use_float=True):
                doc_id = _doc_id(doc)
                corrected = seen_ids.get(doc_id)
                if corrected is not None:
                    _set_laufnummer(doc, corrected)
                if not first_written:
                    fout.write(",")
                json.dump(doc, fout)
                first_written = False
            fout.write("]")

        file_summary_rows.append(
            {
                "file": path.name,
                "n_docs": n_docs,
                "n_new_docs": len(new_meta),
                "n_tag_wechsel": n_transitions + (1 if i == 0 else 0),
                "corrected_counter_after": corrected_counter,
                "output_path": str(out_path),
            }
        )
        print(
            f"[correct_session_folder] {path.name}: {n_docs} Dokumente, {len(new_meta)} neu, "
            f"{n_transitions + (1 if i == 0 else 0)} Tag-Wechsel, Zaehler danach = "
            f"{corrected_counter} -> {out_path}"
        )

    file_summary = pd.DataFrame(file_summary_rows)

    group_counts = Counter(seen_ids.values())
    group_rows = [{"laufnummer": tag, "n_docs": n} for tag, n in sorted(group_counts.items())]
    group_summary = pd.DataFrame(group_rows)
    if not group_summary.empty:
        median_n = group_summary["n_docs"].median()
        group_summary["anomalous"] = group_summary["n_docs"] > 1.7 * median_n
        n_anom = int(group_summary["anomalous"].sum())
        if n_anom:
            anom_tags = group_summary.loc[group_summary["anomalous"], "laufnummer"].tolist()
            warnings.warn(
                f"{input_dir}: {n_anom} korrigierte Laufnummer-Gruppe(n) deutlich groesser als "
                f"der Median ({median_n:.0f} Dokumente) -- moeglicher nicht automatisch "
                f"trennbarer Merge (z.B. wie im bekannten C45_y-Tag-28-Fall): {anom_tags}. "
                f"Zur manuellen Pruefung markiert, NICHT automatisch aufgeteilt."
            )

    return file_summary, group_summary


def split_by_laufnummer(prep_dir: PathLike, output_dir: PathLike) -> pd.DataFrame:
    """
    Liest ALLE korrigierten LevNNN-Dateien aus prep_dir (nicht nur die letzte -- einige
    Sessions sind NICHT durchgehend kumulativ, siehe unten) und schreibt je korrigierter
    Laufnummer EINE JSON-Datei nach output_dir (gleiche Array-of-Documents-Struktur wie die
    Quelldateien, nur auf die jeweilige Laufnummer gefiltert). Dokumente werden ueber alle
    Dateien hinweg per _id.$oid dedupliziert (erstes Auftreten gewinnt) -- unproblematisch,
    da correct_session_folder() sicherstellt, dass ein Dokument in jeder Datei, in der es
    vorkommt, dieselbe korrigierte Laufnummer traegt.

    WICHTIG: urspruenglich wurde nur die hoechstnummerierte (vermeintlich kumulative) Datei
    gelesen, das war fuer 20260630_ALU_y und 20260701_C45_y korrekt (dort ist die letzte Datei
    nachweislich die Vereinigung aller Dokumente der Session), aber falsch fuer
    20260701_ALU_x: dort bricht die Kumulativitaet zweimal (vor Lev011 und vor Lev014) --
    die letzte Datei (Lev019) enthaelt nur 996 der insgesamt 2047 Dokumente der Session.
    Das Einlesen aller Dateien mit Deduplizierung ist robust gegen diesen Fall.

    Ein Durchlauf ueber alle Dateien: je neu gesehenem Dokument wird die (bereits korrigierte)
    Laufnummer gelesen und das Dokument unveraendert an die passende Ausgabedatei angehaengt
    (Datei wird beim ersten Auftreten dieser Laufnummer lazy geoeffnet und bis zum Ende offen
    gehalten -- bei ueblicherweise nur 10-40 verschiedenen Laufnummern je Session unproblematisch).

    Gibt eine Zusammenfassungstabelle zurueck (eine Zeile je Laufnummer: n_docs, output_path).
    """
    prep_dir = Path(prep_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_lev_files(prep_dir)

    handles: Dict[int, TextIO] = {}
    is_first: Dict[int, bool] = {}
    counts: Counter = Counter()
    seen_ids: set = set()
    n_missing = 0
    n_total_docs_scanned = 0

    def _handle_for(tag: int):
        fh = handles.get(tag)
        if fh is None:
            out_path = output_dir / f"axon_logger_ptw_Laufnummer{tag:03d}.raw.json"
            fh = open(out_path, "w", encoding="utf-8")
            fh.write("[")
            handles[tag] = fh
            is_first[tag] = True
        return fh

    try:
        for path in files:
            with open(path, "rb") as fin:
                for doc in ijson.items(fin, "item", use_float=True):
                    n_total_docs_scanned += 1
                    doc_id = _doc_id(doc)
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    tag = _get_laufnummer(doc)
                    if tag is None:
                        n_missing += 1
                        continue
                    fh = _handle_for(tag)
                    if not is_first[tag]:
                        fh.write(",")
                    json.dump(doc, fh)
                    is_first[tag] = False
                    counts[tag] += 1
    finally:
        for fh in handles.values():
            fh.write("]")
            fh.close()

    if n_missing:
        warnings.warn(
            f"{prep_dir}: {n_missing} eindeutige Dokumente ohne lesbare Laufnummer wurden "
            f"beim Aufteilen uebersprungen (nicht in separated_by_ln enthalten)."
        )

    rows = [
        {
            "laufnummer": tag,
            "n_docs": n,
            "output_path": str(output_dir / f"axon_logger_ptw_Laufnummer{tag:03d}.raw.json"),
        }
        for tag, n in sorted(counts.items())
    ]
    summary = pd.DataFrame(rows)
    print(
        f"[split_by_laufnummer] {prep_dir} ({len(files)} Dateien, {n_total_docs_scanned} "
        f"Dokumente gescannt, {len(seen_ids)} eindeutig) -> {len(summary)} Laufnummer-Dateien "
        f"in {output_dir} ({sum(counts.values())} Dokumente, {n_missing} ohne Laufnummer "
        f"uebersprungen)"
    )
    return summary


if __name__ == "__main__":
    DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "Hermle"
    SESSIONS = ["20260630_ALU_y", "20260701_ALU_x", "20260701_C45_y"]

    for session in SESSIONS:
        session_dir = DATA_ROOT / session
        in_dir, out_dir = resolve_input_output(session_dir)
        print(f"=== {session}: {in_dir} -> {out_dir} ===")
        file_summary, group_summary = correct_session_folder(in_dir, out_dir)

        file_summary_path = out_dir / f"laufnummer_correction_file_summary_{session}.csv"
        group_summary_path = out_dir / f"laufnummer_correction_group_summary_{session}.csv"
        file_summary.to_csv(file_summary_path, index=False)
        group_summary.to_csv(group_summary_path, index=False)
        print(f"Zusammenfassungen gespeichert: {file_summary_path}, {group_summary_path}\n")

    for session in SESSIONS:
        session_dir = DATA_ROOT / session
        prep_dir = session_dir / "prep"
        split_dir = session_dir / "separated_by_ln"
        print(f"=== {session}: {prep_dir} -> {split_dir} ===")
        split_summary = split_by_laufnummer(prep_dir, split_dir)
        split_summary_path = split_dir / f"laufnummer_split_summary_{session}.csv"
        split_summary.to_csv(split_summary_path, index=False)
        print(f"Zusammenfassung gespeichert: {split_summary_path}\n")
