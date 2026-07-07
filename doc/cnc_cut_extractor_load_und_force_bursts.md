# `CNCCutExtractor` — Erklärung ausgewählter Methoden

Bezieht sich auf `py/src/cnc_cut_extractor.py`.

> **Visuelle Flussdiagramme** der drei Segmentierungsstrategien (Knoten pro
> Funktion, farblich nach Funktion/Filter/Diagnose/Ergebnis-CSV codiert):
> [CNC-Schnitt-Pipeline — Segmentierungsstrategien](https://claude.ai/code/artifact/fc592f94-6a36-48f3-84a3-2d348eff6d61)

## Übersicht: untersuchte Segmentierungsstrategien

Die Rohaufzeichnung (kontinuierlicher `axon_logger`-Export) muss in
sinnvolle Analyseeinheiten (Blöcke/Schnitte/Level) zerlegt werden. Dafür
wurden drei unterschiedliche Strategien umgesetzt, die alle auf derselben
Rohdatenbasis aufsetzen (`_force_bursts()` für die reine Burst-Erkennung,
`_machine_frame()` für die Achsdaten), sich aber im eigentlichen
Segmentierungskriterium unterscheiden. Alle drei besitzen begleitende
Diagnosefunktionen, die vor dem scharfen Einsatz der jeweiligen Schwellen
im Notebook geprüft werden sollen (Muster: erst *_diagnostics(), dann
detect_*()/export_*()).

### 1. Zeit-/Richtungsumkehr-basierte Segmentierung (Standardpipeline)

Zwei-stufiges Verfahren: erst Bearbeitungs-**Blöcke** über die Zeit
abgrenzen, dann innerhalb jedes Blocks die einzelnen **Schnitte** über
Richtungsumkehr der Vorschubachse trennen.

- **Blockebene:** `detect_blocks()` filtert die rohen Force-Bursts aus
  `_force_bursts()` nach Mindestdauer (`min_block_duration_s`), optional
  kombiniert mit Amplitude (`_burst_amplitude()`) und Programmlaufanteil
  (`_programrunning_frac()`), um kurze "Heartbeat"-Bursts von echten
  Bearbeitungsblöcken zu unterscheiden.
- **Schnittebene:** `_cut_segments_raw()` erkennt die Richtungsumkehr der
  per `cut_axis_index` vorgegebenen Vorschubachse (Peak-Erkennung auf der
  geglätteten Position); `detect_cuts_in_block()` wendet darauf zusätzlich
  optionale Filter an (`max_cut_velocity`, `cut_direction_sign`,
  `max_other_axis_range`), um Rückläufe/Leerfahrten zu verwerfen.
- **Extraktion + Export:** `extract_cut()` je erkanntem Schnitt, orchestriert
  durch `run()` (schreibt je Schnitt ein CSV plus Radialkraft-Winkelprofil).
- **Kalibrier-Diagnose:** `burst_diagnostics()` (Block-Ebene, vor
  `min_burst_amplitude`/`min_programrunning_frac`), `cut_diagnostics()`
  (Schnitt-Ebene, vor `max_cut_velocity`/`cut_direction_sign`/
  `max_other_axis_range`).

### 2. Lückenfreie Aufnahme-Fenster ("gap-free segments")

Alternative, deutlich einfachere Segmentierung ohne Richtungsumkehr-Logik.
Grundlage ist eine Analyse des NC-Programms: die Aufzeichnung läuft laut
NC-Programm nur, *während* tatsächlich geschnitten wird, und wird danach
abgeschaltet — jeder ununterbrochene Force-Burst entspricht daher bereits
einer echten, in sich abgeschlossenen Aufzeichnungssitzung.

- `gap_free_segments()`: dünner öffentlicher Wrapper um `_force_bursts()` —
  liefert **alle** Bursts ungefiltert (auch kurze Heartbeats), ohne
  Dauer-/Amplitudenschwelle.
- `export_gap_free_segments()`: exportiert jeden Burst als eigene CSV
  (Rohsignal-Format wie `extract_cut().raw_table`, aber ohne Winkel-/
  Radialkraft-Zusatzberechnung).

### 3. Z-Tiefen-basierte Level-Gruppierung

Alternative **Gruppierung** (kein Ersatz für die Schnitt-Erkennung, setzt
auf den Blöcken aus `detect_blocks()` auf) für den Standzeitversuch: die
axiale Schnitttiefe (`ap`, R104) wird von Hand zwischen Versuchen geändert
und bleibt für 8 Schnitte (= 2 Blöcke) konstant, während Drehzahl/Vorschub
bereits alle 4 Schnitte (= 1 Block) wechseln. Hier wird also nicht nach
Zeit, sondern nach Z-Position gruppiert.

- `z_level_diagnostics()`: gruppiert die Blöcke aus `detect_blocks()` anhand
  des Z-Median der Vorschubachse zu Leveln, erkennt dabei bekannte
  Park-/Sicherheitshöhen (`z_safe_heights`, z.B. 10/20 mm) als
  `near_safe_height` (kein echter Schnitt).
- `export_z_level_segments()`: fasst je Level alle zugehörigen Blöcke zu
  einer gemeinsamen CSV zusammen, verwirft dabei Blöcke auf
  Park-/Eilganghöhe (`z_median > 0`).

**Kurzreferenz:**

| Strategie | Segmentierungskriterium | Detect-/Group-Funktion | Export-Funktion | Diagnose-Funktion |
|---|---|---|---|---|
| Zeit + Richtungsumkehr | Burst-Dauer/-Amplitude, dann Richtungsumkehr der Vorschubachse | `detect_blocks()` + `detect_cuts_in_block()` (nutzt `_cut_segments_raw()`) | `run()` (via `extract_cut()`) | `burst_diagnostics()`, `cut_diagnostics()` |
| Lückenfreie Fenster | Zusammenhängender Force-Burst = 1 Aufzeichnungssitzung | `gap_free_segments()` (nutzt `_force_bursts()`) | `export_gap_free_segments()` (via `extract_cut()`) | — (keine eigene, Bursts sind ungefiltert) |
| Z-Tiefen-Level | Z-Median über mehrere Blöcke hinweg konstant | `z_level_diagnostics()` (gruppiert `detect_blocks()`) | `export_z_level_segments()` (via `extract_cut()`) | `z_level_diagnostics()` selbst (liefert Diagnosespalten, keine separate Funktion) |

---

## `_to_native(value)` (Zeile 92–109, Modul-Funktion)

**Zweck:** Rekursive Hilfsfunktion, die MongoDB-Extended-JSON-Sondertypen
(z.B. `{"$numberLong": "123"}`, `{"$date": "..."}`, `{"$oid": "..."}`,
`Decimal128`) in native Python-/NumPy-Typen umwandelt. Der axon_logger-Export
enthält solche Wrapper-Objekte, weil MongoDB Zahlen/Zeitstempel beim
JSON-Export nicht als reine `int`/`float`/`str` schreibt, sondern typisiert
einbettet.

**Ablauf:**

1. `Decimal` (Python-Dezimalzahl, aus `decimal.Decimal`) → `float`
   (Zeile 93–94).
2. `dict`: Prüft der Reihe nach auf die bekannten Extended-JSON-Schlüssel
   `$numberLong`, `$numberDouble`, `$numberDecimal`, `$date`, `$oid`
   (Zeile 96–105) und gibt den entpackten Wert zurück. Ist keiner davon
   vorhanden, wird das Dict rekursiv mit `_to_native` auf jeden Wert
   angewendet (Zeile 106) — für "normale" verschachtelte Dicts ohne
   Sondertyp.
3. `list`: rekursiv jedes Element konvertieren (Zeile 107–108).
4. Alles andere (bereits nativ, z.B. `float`, `int`, `str`, `bool`): unverändert
   zurückgeben (Zeile 109).

**Verwendung:** Wird u.a. in `_machine_frame()` auf die Achswerte
(`currentact`, `positionact`) angewendet, da diese je nach Dokument als
native Zahl oder als `$numberDouble`/`$numberDecimal`-Wrapper vorliegen
können.

---

## `CutResult` (Zeile 112–125, `@dataclass`)

**Zweck:** Reiner Datencontainer für das Ergebnis eines einzelnen
extrahierten Schnitts — wird von `extract_cut()` erzeugt und in
`self.results` gesammelt.

**Felder:**

- `block_id`, `cut_id`: Nummerierung (Block- bzw. Schnittindex).
- `t_start`, `t_end`: Zeitfenster des Schnitts (relativ zu `t_ref`).
- `raw_table`: die SinuTrace-Stil-Rohsignaltabelle (`pd.DataFrame`).
- `angle_deg`: hochinterpolierte Spindelposition (0–360°) als
  `np.ndarray`.
- `force_ds`: Dict der herunterinterpolierten Kraftkanäle in Newton
  (`{"force_0": array, ...}`).
- `radial_force_profile`: `pd.DataFrame` mit der Summe/dem Mittel der
  Radialkraft je Winkel-Bin.
- `csv_path`: Pfad der geschriebenen CSV-Datei, `None` solange noch nicht
  exportiert (wird erst in `run()` gesetzt).
- `warnings`: Liste von Warnhinweisen, die während der Extraktion
  aufgetreten sind (`field(default_factory=list)`, damit jede Instanz
  ihre eigene leere Liste bekommt statt eine geteilte Default-Liste).

---

## `__init__(self, json_path, output_dir="./schnitte", ...)` (Zeile 249–336)

**Zweck:** Konstruktor der Pipeline — validiert die Konfiguration und legt
den initialen (leeren) Zustand an. Die Bedeutung der einzelnen Parameter ist
ausführlich im Klassen-Docstring (Zeile 136–245) erklärt; hier geht es um
die Validierungslogik im Code selbst.

**Wichtige Validierungen (schlagen mit `ValueError` fehl, falls verletzt):**

1. **`cut_axis_index` ist Pflicht** (Zeile 277–282): `None` wird explizit
   abgelehnt — es gibt bewusst keine automatische Achsauswahl mehr (siehe
   Klassen-Docstring: das führte bei reinen Y-/X-/Z-Schnitten zu
   Fehlerkennung durch Rauschen/Restbewegung einer anderen Achse).
2. **`cut_axis_index` muss in `feed_axis_indices` enthalten sein**
   (Zeile 283–287).
3. **`cut_direction_sign`** (Zeile 289–298): darf nur `-1`, `+1` oder
   `None` sein; ergibt ohne gesetzte `cut_axis_index` keinen Sinn (wird
   defensiv nochmal geprüft, obwohl `cut_axis_index` an dieser Stelle
   durch Punkt 1 bereits gesetzt sein muss).
4. **`min_programrunning_frac`** (Zeile 307–310): muss im Intervall
   `[0, 1]` liegen (ist ein Anteil).
5. **`max_cut_velocity`** (Zeile 316–319): falls gesetzt, muss `> 0` sein.
6. **`max_other_axis_range`** (Zeile 321–324): falls gesetzt, muss `>= 0`
   sein.

**Weiterer Ablauf:**

- Alle Konstruktor-Parameter werden 1:1 als Instanzattribute gespeichert
  (`self.json_path`, `self.output_dir`, usw.).
- `feed_axis_indices` wird zu einem Tuple gemacht (Zeile 276) — unveränderlich,
  da es als Schlüssel-/Vergleichswert in mehreren Methoden verwendet wird.
- `os.makedirs(self.output_dir, exist_ok=True)` (Zeile 328): legt das
  Ausgabeverzeichnis an, falls es noch nicht existiert.
- **Leerer Ladezustand** (Zeile 330–336): `self._force_docs` wird mit den
  drei Kanal-Keys `{0: [], 1: [], 2: []}` vorinitialisiert,
  `self._machine_docs = []`, `self._t_ref = None` — diese werden erst von
  `load()` befüllt. `self.results` (Liste der `CutResult`) und
  `self.summary_` (Zusammenfassungs-DataFrame) starten ebenfalls leer/`None`
  und werden erst von `run()` gefüllt.

---

## `_rel_t(self, ts)` (Zeile 397–398)

**Zweck:** Kleine, aber zentrale Hilfsfunktion — wandelt einen absoluten
`pd.Timestamp` in eine relative Zeit in Sekunden um, bezogen auf den in
`load()` ermittelten globalen Nullpunkt `self._t_ref`:

```python
def _rel_t(self, ts: pd.Timestamp) -> float:
    return (ts - self._t_ref).total_seconds()
```

Wird von praktisch allen zeitbasierten Methoden verwendet (`_force_bursts`,
`_concat_force_channel`, `_machine_frame`, `sampling_diagnostics*`, ...),
damit intern durchgängig mit einfachen `float`-Sekunden statt
`pd.Timestamp`-Objekten gerechnet werden kann.

---

## `_machine_frame(self, t_start, t_end)` (Zeile 459–500)

**Zweck:** Baut aus den geladenen `machine`-Dokumenten, die das Fenster
`[t_start, t_end]` (mit Rand) überlappen, einen 250-Hz-`DataFrame` mit
Spindel- und Vorschubachsen-Signalen sowie den GCode-Status-Flags. Wird von
sehr vielen anderen Methoden verwendet (`_cut_segments_raw`,
`_programrunning_frac`, `extract_cut`, `z_level_diagnostics`, ...) — die
zentrale Stelle, an der aus den rohen `machine`-Snapshots eine tabellarische
Zeitreihe wird.

**Ablauf:**

1. **Relevante Achsen bestimmen** (Zeile 465): Spindelachse
   (`spindle_axis_index`) plus alle Vorschubachsen (`feed_axis_indices`).
2. **Dokumente filtern und entpacken** (Zeile 467–490): Für jedes
   `machine`-Dokument wird geprüft, ob es das Fenster überlappt (mit 0.5 s
   Rand, Zeile 471 — großzügiger als die 10 ms bei den Kraftkanälen, da
   `machine`-Dokumente typischerweise 1 s Snapshots bündeln). Überlappt es,
   wird für **jeden** der 250 Snapshots im Dokument eine Zeile gebaut mit:
   - Zeitstempel `t` (Snapshot-Index / 250 Hz)
   - Status-Flags (`programrunning`, `ncstart`, `ncstop`, `machineworking`,
     `programstopped`, `reset`)
   - für jede interessierende Achse: `axis{i}_currentact`,
     `axis{i}_positionact`, `axis{i}_stop`
3. **Leerfall** (Zeile 491–492): Keine passenden Zeilen → leerer
   `DataFrame`.
4. **Sortieren** (Zeile 493): Nach Zeit `t`, Index zurückgesetzt.
5. **Typkonvertierung erzwingen** (Zeile 494–499): `currentact`/
   `positionact` werden explizit mit `pd.to_numeric(..., errors="coerce")`
   in Float konvertiert — unabhängig davon, ob im JSON überall native
   Floats oder (teilweise) Mongo-Extended-JSON-Zahlentypen vorlagen. Das
   verhindert, dass beim CSV-Export einzelne Zellen als Text-/
   Objekt-Repräsentation landen (z.B. `"12.3"` statt `12.3`, wenn eine
   Zeile zufällig noch einen `$numberDecimal`-String enthielt).

**Rückgabe:** `pd.DataFrame`, eine Zeile je 250-Hz-Snapshot.

---

## `machine_frame(self, t_start, t_end)` (Zeile 502–504)

**Zweck:** Rein öffentlicher, dokumentierter Wrapper um `_machine_frame()` —
macht die interne Achsdaten-Tabelle für Diagnose-Plots im Notebook
zugänglich, ohne die interne Methode direkt anzusprechen (Konvention:
Unterstrich-Methoden gelten als privat/intern).

---

## `concat_force_channel(self, sensoridx, t_start, t_end)` (Zeile 506–509)

**Zweck:** Analog zu `machine_frame()` — öffentlicher Wrapper um
`_concat_force_channel()`, gedacht für Diagnose-Plots im Notebook (z.B. um
einen Kraftkanal über ein beliebiges Zeitfenster zu plotten, ohne einen
kompletten Schnitt zu extrahieren).

---

## `data_time_range(self)` (Zeile 511–526)

**Zweck:** Liefert den gesamten Zeitbereich (relativ zu `t_ref`, in
Sekunden), der von **allen** geladenen `force`- und `machine`-Dokumenten
abgedeckt wird — unabhängig von der Block-/Schnitt-Erkennung (die kurze
Heartbeat-Bursts am Rand ggf. verwirft). Nützlich, um z.B. "die letzten N
Sekunden der Datei" ohne vorherige Segmentierung zu plotten.

**Ablauf:**

1. Für jeden Force-Kanal (alle drei `sensoridx`) und jedes Dokument wird
   das Ende (`t0 + n/20000`) berechnet und gesammelt (Zeile 519–521).
2. Für jedes `machine`-Dokument analog das Ende (`t0 + len(data)/250`)
   (Zeile 522–523).
3. Keine Dokumente geladen → `ValueError` ("erst `load()` aufrufen",
   Zeile 524–525).
4. Rückgabe `(0.0, max(end_times))` (Zeile 526) — Start ist per Definition
   `0.0` (relativ zu `t_ref`, dem frühesten gesehenen Zeitstempel über
   *alle* Dokumente), Ende ist der späteste Endzeitpunkt über alle Kanäle.

**Rückgabe:** `(t_min, t_max)` als Tupel von Sekunden.

---

## `_burst_amplitude(self, t_start, t_end)` (Zeile 528–544)

**Zweck:** Berechnet die maximale Peak-to-Peak-Amplitude (rohe ADC-Counts)
über alle drei Kraftkanäle (`FORCE_SENSOR_INDICES`) innerhalb eines
Zeitfensters — Kennzahl, um zu unterscheiden, ob in einem Fenster
tatsächlich Zerspankraft anliegt oder nur Rauschen vorliegt.

**Ablauf:**

1. Für jeden Kraftkanal (Zeile 535–543):
   - Rohdaten holen via `_concat_force_channel(si, t_start, t_end)`
     (Zeile 536).
   - Keine Daten im Kanal → überspringen (Zeile 537–538).
   - Exakte Maskierung auf `[t_start, t_end]` (Zeile 539–540) — analog zum
     wiederkehrenden Muster in `_programrunning_frac`/`extract_cut`, da
     `_concat_force_channel` auch leicht überlappende Dokumente zurückgibt.
   - `np.ptp(raw)` (Peak-to-Peak, Max − Min) berechnen und sammeln
     (Zeile 543).
2. Rückgabe: größter Wert über alle Kanäle, oder `0.0`, falls kein Kanal
   Daten im Fenster hatte (Zeile 544).

**Verwendung:** Zentrale Kennzahl in `detect_blocks()` (Amplitudenfilter
`min_burst_amplitude`) und `burst_diagnostics()`/`cut_diagnostics()` (als
Diagnosespalte `amplitude_ptp`/`force_amplitude`).

---

## `load(self)` (Zeile 341–395)

Erster Schritt der Pipeline: liest den rohen MongoDB-JSON-Export einmal komplett ein
und hält davon nur das, was später gebraucht wird, im Speicher
(`self._force_docs`, `self._machine_docs`).

**Ablauf im Detail:**

1. **Streaming-Parsing statt `json.load`** (Zeile 351–352)
   Die Datei wird mit `ijson.items(f, "item")` geöffnet und Dokument für Dokument
   aus dem JSON-Array gestreamt, statt die komplette Datei auf einmal zu parsen.
   Das ist nötig, weil die Exporte 300+ MB groß sind — ein normales `json.load`
   würde alles gleichzeitig in den Speicher laden.

2. **Filterung nach Sensortyp** (Zeile 353–355)
   Für jedes Dokument wird `sensortype` geprüft. Nur `"force"` und `"machine"`
   werden behalten; `"acceleration"` und `"distance"` werden komplett verworfen
   (`continue`), weil sie für die Aufgabenstellung nicht gebraucht werden —
   spart Speicher und Zeit.

3. **Zeitstempel sammeln** (Zeile 357–358)
   Aus `doc["time"]["$date"]` wird ein `pd.Timestamp` gebaut und in `times_seen`
   gesammelt. Das dient später dazu, einen globalen Zeit-Nullpunkt (`self._t_ref`)
   zu bestimmen.

4. **Force-Dokumente ablegen** (Zeile 360–369)
   Bei `sensortype == "force"` wird das Dokument nach `sensoridx` (0, 1 oder 2 —
   die drei Kraft-Kanäle der Kistler-Messplatte) in `self._force_docs[idx]`
   einsortiert. Gespeichert werden:
   - `t0`: Start-Zeitstempel
   - `n`: Anzahl der Rohdatenpunkte (typischerweise 20000 bei 20 kHz)
   - `data`: die Rohdaten als NumPy-Array (`float64`)
   - `process`: Prozess-Flag aus dem Dokument

5. **Machine-Dokumente ablegen** (Zeile 370–377)
   Bei `sensortype == "machine"` wird das Dokument in `self._machine_docs`
   gehängt, mit `t0`, den rohen `data` (Liste von 250 Snapshots/Sekunde mit
   Achspositionen etc. — bleibt bewusst als Liste von dicts, wird erst später
   bei Bedarf in `_to_native` konvertiert) und `process`.

6. **Validierung** (Zeile 379–382)
   Wurden gar keine `force`- oder `machine`-Dokumente gefunden, wird ein
   `ValueError` geworfen — sonst würde die Pipeline später mit leeren Daten
   stillschweigend falsche Ergebnisse produzieren.

7. **Referenzzeitpunkt setzen** (Zeile 383)
   `self._t_ref = min(times_seen)` — der früheste gesehene Zeitstempel wird als
   globaler Nullpunkt festgelegt. Alle späteren Zeitberechnungen (`_rel_t`,
   Zeile 397–398) laufen relativ dazu in Sekunden, nicht mehr als absolute
   Timestamps.

8. **Sortierung nach Zeit** (Zeile 385–387)
   Sowohl die drei Force-Kanäle als auch die Machine-Dokumente werden nach `t0`
   sortiert. Das ist wichtig, weil der JSON-Export nicht zwingend zeitlich
   geordnet sein muss (MongoDB-Dump), spätere Methoden (z.B. `_force_bursts`,
   Zeile 403 ff.) aber auf aufsteigende Zeitreihen angewiesen sind, um
   zusammenhängende Bursts zu erkennen.

9. **Diagnose-Ausgabe** (Zeile 389–394)
   Ein `print` gibt aus, wie viele Dokumente je Force-Kanal geladen wurden, wie
   viele Machine-Dokumente, und den ermittelten `t_ref` — nützlich, um im
   Notebook schnell zu prüfen, ob das Parsing plausibel war (z.B. ob alle drei
   Kraftkanäle ungefähr gleich viele Dokumente haben).

10. **Return `self`** (Zeile 395)
    Die Methode gibt die Instanz selbst zurück, damit man Method-Chaining machen
    kann, z.B. `extractor = CNCCutExtractor(...).load()`.

**Kurz gesagt:** `load()` ist der reine I/O-/Parsing-Schritt — er liest die
riesige JSON-Datei speicherschonend ein, filtert auf die relevanten Kanäle,
sortiert sie zeitlich und legt die Basis (`_force_docs`, `_machine_docs`,
`_t_ref`) für alle nachfolgenden Verarbeitungsschritte (Burst-Erkennung,
Schnitt-Segmentierung, CSV-Export) in der Pipeline.

---

## `_force_bursts(self)` (Zeile 403–429)

**Zweck:** Findet zusammenhängende Aufnahmefenster ("Bursts") im
Kraft-Referenzkanal — das sind Kandidaten für Bearbeitungsblöcke.

**Ablauf:**

1. **Referenzkanal wählen** (Zeile 410–412): Es wird nur `force_0`
   (Kanal `sensoridx == 0`) angeschaut, nicht alle drei Kanäle — er dient als
   Stellvertreter für "wann wurde überhaupt geloggt". Ist er leer, Fehler
   ("erst `load()` aufrufen").

2. **Iteration über die (bereits zeitlich sortierten) Force-Dokumente**
   (Zeile 416–428):
   - Für jedes Dokument `d` wird `t_start` (relative Startzeit via `_rel_t`) und
     `t_end = t_start + n/20000` berechnet (Dauer = Anzahl Samples / 20 kHz
     Abtastrate).
   - **Lückenprüfung** (Zeile 420): Wenn kein aktueller Burst offen ist
     (`current is None`) ODER die Lücke zum vorherigen Burst-Ende größer als
     10 ms ist (`t_start - current["t_end"] > 0.01`), wird ein **neuer** Burst
     begonnen. Der alte wird (falls vorhanden) abgeschlossen und der Liste
     `bursts` hinzugefügt.
   - Andernfalls (Dokument schließt nahtlos an, Toleranz 10 ms) wird der
     aktuelle Burst einfach verlängert: `t_end` aktualisiert und das Dokument
     an `current["docs"]` angehängt.
   - Am Ende wird der letzte offene Burst noch angehängt (Zeile 427–428).

3. **Rückgabe:** Liste von Dicts `{"t_start", "t_end", "docs"}` — jeder Eintrag
   ist ein zusammenhängender Aufnahmezeitraum mit den zugehörigen
   Rohdokumenten.

Diese Bursts sind noch keine fertigen "Blöcke" — die eigentliche
Block-Erkennung (mit Mindestdauer, Amplituden-Schwelle etc., siehe
Docstring-Parameter `min_block_duration_s`, `min_burst_amplitude`) passiert in
einer späteren Methode (`detect_blocks`), die diese Rohliste weiterfiltert.

---

## `_concat_force_channel(self, sensoridx, t_start, t_end)` (Zeile 431–454)

**Zweck:** Baut aus den fragmentierten, dokumentweise geloggten Rohdaten eines
einzelnen Kraftkanals eine einzige zusammenhängende `(t, v)`-Zeitreihe für ein
gegebenes Zeitfenster.

**Ablauf:**

1. **Kanal auswählen** (Zeile 438): Holt alle Dokumente des gewünschten
   `sensoridx` (0, 1 oder 2).

2. **Relevante Dokumente filtern** (Zeile 440–448): Für jedes Dokument wird
   sein Zeitintervall `[d_t0, d_t1]` berechnet. Überlappt es **nicht** mit dem
   angefragten Fenster `[t_start, t_end]` (mit 10 ms Takttoleranz, Zeile 444),
   wird es übersprungen. Andernfalls wird pro Sample eine Zeitachse erzeugt
   (`d_t0 + np.arange(n)/20000`) und sowohl Zeit- als auch Wertarray in Listen
   gesammelt.

3. **Leerfall** (Zeile 449–450): Kein passendes Dokument gefunden → leere
   Arrays zurückgeben.

4. **Konkatenieren & sortieren** (Zeile 451–454): Alle gesammelten
   Zeit-/Wert-Segmente werden zu einem Array zusammengefügt
   (`np.concatenate`) und nach Zeit sortiert (`np.argsort`), falls Dokumente
   nicht exakt in der richtigen Reihenfolge lagen.

**Rückgabe:** `(t, v)` — Zeitstempel (Sekunden relativ zu `t_ref`) und
Rohwerte (ADC-Counts) des jeweiligen Kraftkanals, exakt zugeschnitten auf das
angefragte Fenster. Diese Funktion wird pro Schnitt (`t_start`/`t_end` eines
erkannten Cuts) für jeden der drei Kraftkanäle aufgerufen, um die Rohsignale
fürs CSV/die Weiterverarbeitung (z.B. `MSU.adc_counts_to_force`) zu
extrahieren.

---

## `_programrunning_frac(self, t_start, t_end)` (Zeile 546–563)

**Zweck:** Berechnet, welcher Anteil der `machine`-Snapshots in einem
Zeitfenster `[t_start, t_end]` das Flag `programrunning == True` haben — also
wie "aktiv" die Maschine in diesem Fenster wirklich war. Wird als
Rettungskriterium für kurze Bursts genutzt (siehe `min_programrunning_frac` im
Konstruktor, Zeile 200–204): ein kurzer ~1s-Burst kann trotz Dauer <
`min_block_duration_s` als echter Block gezählt werden, wenn die Maschine
währenddessen überwiegend im Programmlauf war.

**Ablauf:**

1. **Machine-Frame holen** (Zeile 557): Ruft `_machine_frame(t_start, t_end)`
   auf (Zeile 459 ff.), die einen 250-Hz-DataFrame mit Achsdaten und
   Status-Flags für alle `machine`-Dokumente baut, die das Fenster
   überlappen.

2. **Leerfall** (Zeile 558–559): Keine Machine-Daten im Fenster → `0.0`.

3. **Exakte Maskierung** (Zeile 560): Das ist der wichtigste Schritt, laut
   Docstring explizit begründet: `_machine_frame()` arbeitet dokumentweise und
   puffert dabei bis zu 0.5 s über `[t_start, t_end]` hinaus (weil es ganze
   Dokumente nimmt, die das Fenster nur überlappen müssen, Zeile 471). Ohne
   diese Nachmaskierung würden bei kurzen Bursts (~1 s Heartbeat) Snapshots
   aus dem benachbarten Block oder Leerlauf mit reinrutschen und den
   Mittelwert verfälschen. Deshalb wird hier zusätzlich strikt auf
   `t >= t_start` und `t <= t_end` gefiltert — analog zur gleichen
   Maskierungslogik in `extract_cut()`.

4. **Erneuter Leerfall** (Zeile 561–562): Bleibt nach der exakten Maskierung
   nichts übrig → `0.0`.

5. **Mittelwert bilden** (Zeile 563): `mdf["programrunning"].mean()` — da die
   Spalte boolesch ist, ergibt der Mittelwert direkt den Anteil an
   `True`-Werten (z.B. 0.8 = 80 % der Snapshots im Fenster hatten
   `programrunning == True`).

**Rückgabe:** `float` zwischen 0.0 und 1.0.

Diese Funktion wird in `detect_blocks()` (Zeile 569 ff.) zusammen mit
`_burst_amplitude()` verwendet, um zu entscheiden, ob ein kurzer Burst
trotzdem als Bearbeitungsblock gilt.

---

## `detect_blocks(self)` (Zeile 569–620)

**Zweck:** Filtert aus allen rohen Force-Bursts (`_force_bursts()`) diejenigen
heraus, die tatsächlich als **Bearbeitungsblöcke** gelten sollen — jeder
verbleibende Block entspricht normalerweise `cuts_per_block` (i.d.R. 4)
Schnitten.

**Zwei Modi, je nachdem ob `min_burst_amplitude` gesetzt ist:**

1. **Reine Dauer-Heuristik** (Default, `min_burst_amplitude is None`,
   Zeile 587–594):
   Ein Burst wird als Block gewertet, wenn seine Dauer
   `>= min_block_duration_s` ist. Kurze Heartbeat-Bursts (~1s ohne echte
   Bearbeitung) fallen raus. Am Ende ein `print` mit Anzahl gefundener vs.
   akzeptierter Bursts.

2. **Kombinierte Amplitude+Dauer/programrunning-Logik** (Zeile 596–619),
   aktiv sobald `min_burst_amplitude` gesetzt ist. Für jeden Burst `b`:
   - `duration_ok`: Dauer `>= min_block_duration_s`
   - `amp_ok`: `_burst_amplitude(t_start, t_end) >= min_burst_amplitude`
     (Zeile 602)
   - **Amplitude zu niedrig** (Zeile 603–606): Burst wird verworfen
     (vermutlich Rauschen). Falls er dabei eigentlich lang genug gewesen
     wäre, wird das in `n_rejected_noise` gezählt (Diagnose: "langer Burst
     aber nur Rauschen").
   - **Amplitude ok UND lang genug** (Zeile 607–609): Burst wird direkt
     akzeptiert.
   - **Amplitude ok, aber zu kurz** (Zeile 610–612): Wird trotzdem "gerettet",
     falls `_programrunning_frac(...) >= min_programrunning_frac` — also die
     Maschine im Fenster überwiegend im Programmlauf war. Zählt in
     `n_recovered_short`.
   - Am Ende ein detailliertes `print` mit Gesamtzahl Bursts, akzeptierten
     Blöcken, verworfenen "Rauschen"-Bursts und geretteten Kurz-Bursts.

**Kernidee:** Mit `min_burst_amplitude=None` verhält sich die Funktion exakt
wie die einfache Dauer-Filterung (Rückwärtskompatibilität, siehe Docstring
Zeile 582–583). Erst wenn man explizit eine Amplitudenschwelle setzt, kommt
die feinere Logik zum Tragen, die zwischen "langem Rauschen" und "kurzem
echten Schnitt" unterscheiden kann.

**Rückgabe:** Liste der akzeptierten Burst-Dicts
(`{"t_start", "t_end", "docs"}`).

---

## `burst_diagnostics(self)` (Zeile 622–655)

**Zweck:** Reine **Diagnose-/Explorationsfunktion** fürs Notebook — zeigt für
**alle** rohen Force-Bursts (nicht nur die aktuell akzeptierten) die
relevanten Kennzahlen in einer Tabelle, damit man vor dem scharfen Setzen von
`min_burst_amplitude`/`min_programrunning_frac` visuell prüfen kann, ob die
Schwellen sinnvoll sind.

**Ablauf:**

1. Holt alle rohen Bursts via `_force_bursts()` (Zeile 635).
2. Für jeden Burst (durchnummeriert `burst_id` ab 1, Zeile 637):
   - `dur`: Dauer
   - `amp`: `_burst_amplitude(t0, t1)` — maximale Peak-to-Peak-Amplitude über
     alle Kraftkanäle
   - `pr_frac`: `_programrunning_frac(t0, t1)` — Anteil
     `programrunning == True`
   - `kept_duration_only`: würde der Burst unter der reinen Dauer-Logik als
     Block gelten?
   - `kept_combined`: würde er unter der **aktuell konfigurierten** Logik
     gelten (Zeile 643–648) — identisch zur Fallunterscheidung in
     `detect_blocks()`, hier aber für **jeden** Burst berechnet, unabhängig
     davon ob er tatsächlich behalten würde.
3. Alle Werte werden als Zeile in einer Liste gesammelt und am Ende als
   `pd.DataFrame` zurückgegeben (Zeile 655).

**Verwendung laut Docstring:** Gedacht zum Vor-Prüfen im Notebook
(Abschnitt 2), *bevor* man `min_burst_amplitude`/`min_programrunning_frac`
scharf schaltet — analog zum Vorgehen bei der Schnitt-Segmentierung
(`cut_diagnostics()`, vgl. Konstruktor-Docstring). Man kann so z.B. sehen:
"Burst 5 hat Amplitude 9 Counts (Rauschen), Burst 6 hat 5000 Counts bei nur
1.2s Dauer, aber `programrunning_frac=0.95`" — und daraus ableiten, welche
Schwellenwerte sinnvoll wären, ohne die Segmentierung schon "scharf" laufen
zu lassen.

---

## `_sampling_channels(self)` (Zeile 657–664)

**Zweck:** Reine Hilfsfunktion, die für die beiden folgenden
Diagnose-Methoden (`sampling_diagnostics()`,
`sampling_diagnostics_in_windows()`) eine einheitliche Sicht auf alle vier
Rohkanäle liefert, damit die Diagnoselogik nicht viermal dupliziert werden
muss.

**Ablauf:**
- Baut ein Dict `{Kanalname: (docs, Abtastrate_Hz, n_fn)}`:
  - `"machine"`: `self._machine_docs`, 250 Hz, `n_fn = len(d["data"])`
    (Anzahl Snapshots je Dokument)
  - `"force_0"`, `"force_1"`, `"force_2"`: je `self._force_docs[idx]`,
    20000 Hz, `n_fn = d["n"]`
- `n_fn` ist nötig, weil sich aus Dokumentanzahl × Rate die Dauer eines
  Dokuments berechnen lässt (`n_fn(d) / rate`).

**Rückgabe:** Dict, das von beiden Sampling-Diagnosefunktionen iteriert wird.

---

## `sampling_diagnostics(self)` (Zeile 666–700)

**Zweck:** Prüft über die **gesamte** Aufzeichnung (nicht auf
Blöcke/Schnitte beschränkt), ob jeder Kanal lückenlos oder in Bursts (mit
Idle-Lücken dazwischen) geloggt wurde — unabhängig von der
Block-/Schnitt-Erkennung.

**Ablauf:**

1. Für jeden Kanal aus `_sampling_channels()` (Zeile 682):
   - Dokumente nach `t0` sortieren (Zeile 685), Start- (`t0s`) und
     Endzeiten (`t1s = t0s + durs`) berechnen (Zeile 686–688).
   - Lücken zwischen aufeinanderfolgenden Dokumenten:
     `gaps = t0s[1:] - t1s[:-1]` (Zeile 689).
   - Toleranz `tol = 2 × Median der Dokumentdauer` (Zeile 690) — toleriert
     Rundungs-/Jitter-Effekte, ohne echte Idle-Lücken zu übersehen.
   - `span` = Gesamtzeitraum von erstem bis letztem Dokument (Zeile 691).
2. Pro Kanal wird eine Zeile mit `n_docs`, `covered_s` (Summe der
   tatsächlich abgedeckten Zeit), `span_s`, `coverage_frac` (Anteil
   abgedeckt vs. Gesamtspanne), `n_gaps` (Anzahl Lücken > Toleranz),
   `max_gap_s`, `mean_gap_s` gesammelt (Zeile 692–699).

**Rückgabe:** `pd.DataFrame`, eine Zeile je Kanal.

**Zweck laut Docstring:** Zeigt, dass sowohl Positions- (250 Hz) als auch
Kraftkanäle (20 kHz) gleichermaßen lückenbehaftet sind — der `axon_logger`
schreibt beide nur in kurzen Bursts, nicht durchgehend.

---

## `sampling_diagnostics_in_windows(self, windows)` (Zeile 702–737)

**Zweck:** Wie `sampling_diagnostics()`, aber je Fenster (z.B. die Blöcke
aus `detect_blocks()`) statt über die gesamte Datei — soll belegen, dass
**innerhalb** eines echten Bearbeitungsblocks keine Lücken auftreten
(`fully_continuous == True`), auch wenn die Gesamtaufzeichnung voller
Lücken ist.

**Ablauf:**

1. Für jedes Fenster `(t_start, t_end)` in `windows` (Zeile 712) und jeden
   Kanal (Zeile 713):
   - Alle Dokumente des Kanals holen, Start-/Endzeiten berechnen
     (Zeile 714–716).
   - Maske: nur Dokumente, die **komplett innerhalb** des Fensters liegen
     (`t0s_all >= t_start-0.01` und `t1s_all <= t_end+0.01`, Zeile 717) —
     anders als `_concat_force_channel`, das auch überlappende Dokumente
     nimmt.
   - Sortieren nach Zeit (Zeile 718–719).
2. **Kein Dokument im Fenster** (Zeile 720–726): Zeile mit `n_docs=0`,
   `coverage_frac=0.0`, `fully_continuous=False`.
3. **Sonst** (Zeile 727–736): Lücken zwischen den gefilterten Dokumenten
   berechnen, Toleranz wie oben (2× Median), `coverage_frac` = abgedeckte
   Zeit / Fensterspanne, `fully_continuous` = größte Lücke `<= tol`.

**Rückgabe:** `pd.DataFrame`, eine Zeile je (Fenster, Kanal)-Kombination.

**Verwendung:** Diagnosewerkzeug, um z.B.
`sampling_diagnostics_in_windows(detect_blocks())` aufzurufen und zu
prüfen, dass jeder erkannte Block tatsächlich lückenlos aufgezeichnet
wurde (Validierung der Block-Erkennung).

---

## `_cut_segments_raw(self, block)` (Zeile 743–829)

**Zweck:** Zentrale, ungefilterte Segmentierungslogik — trennt einen Block
anhand der Richtungsumkehr der per `cut_axis_index` vorgegebenen
Vorschubachse in rohe Schnitt-Segmente. Wird sowohl von
`detect_cuts_in_block()` als auch von `cut_diagnostics()` verwendet, damit
die Segmentierungslogik nur an einer Stelle existiert (Single Source of
Truth).

**Ablauf:**

1. **Machine-Daten holen** (Zeile 769): `_machine_frame(t_start, t_end)`
   für das Blockfenster.
2. **Fallback 1 – keine Machine-Daten** (Zeile 771–777): Wenn `mdf` leer
   ist, wird der Block gleichmäßig in `cuts_per_block` Zeitabschnitte
   geteilt (`np.linspace`), mit Warnmeldung. Die letzten fünf
   Rückgabewerte (`dom_axis`, `axis_range`, `t`, `pos`, `other_axes_pos`)
   sind `None`, da kein Achssignal vorliegt.
3. **Dominante Achse fix vorgegeben** (Zeile 782–789):
   `dom_axis = self.cut_axis_index` (keine automatische Auswahl mehr,
   siehe Konstruktor-Docstring). Position (`pos`) und Zeit (`t`) dieser
   Achse werden extrahiert, `axis_range` = Positions-Spannweite.
   `other_axes_pos` sammelt die Positionsarrays der **übrigen**
   Vorschubachsen (für spätere Nebenachsen-Filterung).
4. **Fallback 2 – zu geringe Bewegung** (Zeile 791–800): Ist
   `axis_range < min_feed_range`, vermutlich reiner Spindel-Leerlauf/
   Rauschen → wieder gleichmäßige Teilung mit Warnung.
5. **Richtungsumkehr-Erkennung** (Zeile 802–811):
   - Position glätten (Rolling Mean, Fenster `reversal_smooth_window`,
     Zeile 802–807).
   - Mindest-Prominenz `prom = axis_range × reversal_prominence_frac`
     (Zeile 808).
   - `find_peaks` auf geglättetem Signal (Maxima) und auf negiertem
     Signal (Minima) → alle Umkehrpunkte (Zeile 809–811).
6. **Segmente aus Umkehrpunkten bilden** (Zeile 814–816): Kanten =
   `[t_start] + Umkehrzeitpunkte + [t_end]`, dedupliziert und sortiert,
   daraus Paare `(t0, t1)`.
7. **Kurzsegment-Merge** (Zeile 819–827): Segmente kürzer als 10 % der
   mittleren Schnittdauer werden an das vorherige Segment angehängt
   (typischerweise eine Restrampe am Blockrand) statt als eigener
   (Mini-)Schnitt gezählt zu werden.

**Rückgabe:** Tupel `(cuts, msgs, dom_axis, axis_range, t, pos,
other_axes_pos)` — in den Fallback-Fällen sind die letzten fünf Werte
`None`.

---

## `detect_cuts_in_block(self, block)` (Zeile 831–915)

**Zweck:** Öffentliche Schnitterkennung — nutzt `_cut_segments_raw()` und
wendet danach optional zusätzliche **Filter** an, um Rückläufe/Leerfahrten
von echten Schnitten zu unterscheiden.

**Ablauf:**

1. Ruft `_cut_segments_raw(block)` auf (Zeile 854).
2. **Filter aktiv?** (Zeile 856–860): Prüft, ob `max_cut_velocity`,
   `cut_direction_sign` oder `max_other_axis_range` gesetzt sind.
3. **Falls ja und ein verlässliches Achssignal vorliegt**
   (`dom_axis is not None`, Zeile 861–905): Für jedes rohe Segment
   `(t0, t1)`:
   - Positionswerte an den Segmentgrenzen per Interpolation
     (`np.interp`, Zeile 869–870).
   - `vel = |p1 - p0| / dur` — mittlere Geschwindigkeit (Zeile 871).
   - `vel_ok`: Geschwindigkeit unter `max_cut_velocity` (falls gesetzt,
     Zeile 872).
   - `dir_ok`: Bewegungsrichtung entspricht `cut_direction_sign` (falls
     gesetzt, Zeile 873).
   - `other_ok`: Spannweite aller **anderen** Vorschubachsen im Segment
     bleibt unter `max_other_axis_range` (falls gesetzt, Zeile 874–879).
   - Nur wenn alle drei Kriterien erfüllt sind, wird das Segment behalten
     (Zeile 882–885).
   - **Sicherheitsnetz** (Zeile 893–905): Würde der Filter **alle**
     Segmente verwerfen, wird er für diesen Block komplett ignoriert und
     die ungefilterten Segmente zurückgegeben (verhindert, dass ein zu
     strenger Filter einen ganzen Block "verschluckt").
4. **Plausibilitätswarnung** (Zeile 907–914): Weicht die Anzahl der (ggf.
   gefilterten) Schnitte von `cuts_per_block` ab, wird gewarnt — Hinweis,
   `reversal_prominence_frac`/`reversal_smooth_window` zu prüfen oder den
   Block manuell zu inspizieren.

**Rückgabe:** `(cuts, msgs)` — Liste der finalen Schnitt-Zeitfenster plus
Warnmeldungen.

---

## `cut_diagnostics(self, block)` (Zeile 917–991)

**Zweck:** Diagnose-Tabelle über **alle rohen** Schnitt-Segmente eines
Blocks (vor jeglicher Filterung) — Pendant zu `burst_diagnostics()`,
gedacht zum Vor-Prüfen im Notebook, bevor `max_cut_velocity`/
`cut_direction_sign`/`max_other_axis_range` scharf geschaltet werden.

**Ablauf:**

1. Holt die rohen Segmente via `_cut_segments_raw(block)` (Zeile 936).
2. Für jedes Segment (Zeile 938–990):
   - `dur`, `amp` (`_burst_amplitude`, nur zur Information/
     Kreuzprüfung, nicht Teil der Filterentscheidung).
   - Falls ein Achssignal vorliegt (`dom_axis is not None and dur > 0`,
     Zeile 941–971):
     - `direction_sign`, `mean_velocity` wie in `detect_cuts_in_block()`.
     - `other_axis_max_range`: größte Spannweite unter den Nebenachsen im
       Segment.
     - `kept_with_velocity_filter` / `kept_with_direction_filter` /
       `kept_with_other_axis_filter`: je `NaN`, falls der entsprechende
       Filter nicht gesetzt ist, sonst `True`/`False`, ob das Segment
       diesen einzelnen Filter bestehen würde.
     - `kept_combined`: UND-Verknüpfung aller **aktiven** Filter
       (Zeile 964–971) — `NaN`, falls kein Filter aktiv ist.
   - Sonst (Fallback-Fall ohne Achssignal, Zeile 972–979): alle
     Filterspalten `NaN`.
3. Sammelt alles in einer Zeile pro Segment (`cut_id` ab 1).

**Rückgabe:** `pd.DataFrame` — erlaubt z.B. zu sehen: "Segment 3 hat
`mean_velocity=45` (über der geplanten Schwelle 10) und würde vom
Geschwindigkeitsfilter verworfen, `direction_sign=-1` passt aber zur
erwarteten Schnittrichtung."

---

## `extract_cut(self, block_id, cut_id, t_start, t_end)` (Zeile 996–1104)

**Zweck:** Das Herzstück der Pipeline — extrahiert für ein gegebenes
Zeitfenster (ein Schnitt) alle Rohsignale, baut die SinuTrace-Stil-
Rohtabelle und berechnet Winkel- sowie Radialkraft-Profil.

**Ablauf:**

1. **Kraftkanäle** (Zeile 1002–1018): Für jeden der drei Kraftkanäle
   (`FORCE_SENSOR_INDICES`):
   - `_concat_force_channel(si, t_start, t_end)` holt die Rohdaten, dann
     exakte Maskierung auf `[t_start, t_end]` (Zeile 1008–1009, gleiches
     Muster wie bei `_programrunning_frac`).
   - Keine Daten → Warnung, Kanal wird übersprungen (Zeile 1010–1012).
   - `MSU.adc_counts_to_force(...)` konvertiert ADC-Rohwerte in Newton
     (Zeile 1014–1016).
2. **Achsdaten** (Zeile 1021–1025): `_machine_frame(t_start, t_end)`,
   ebenfalls exakt auf das Fenster maskiert. Leer → Warnung ("nur
   Kraftkanäle exportiert").
3. **Rohtabelle bauen** (Zeile 1031–1060): Die drei Kraftkanäle haben
   potenziell leicht unterschiedliche Sample-Zeitpunkte (unterschiedliche
   Burst-Dokumente im JSON) — deshalb werden sie per `pd.merge_asof`
   (Toleranz halbe Samplezeit, Zeile 1038–1044) statt einer starren
   Positions-Verknüpfung zusammengeführt. Anschließend werden die
   Achsdaten per `merge_asof` (Toleranz 4 ms, Zeile 1051–1057)
   dazugemerged. Ergebnis: eine gemeinsame Zeittabelle `raw_table`
   (SinuTrace-Stil).
4. **Winkel hochinterpolieren** (Zeile 1063–1070): Aus der (250 Hz)
   Spindelposition wird per `MSU.phi_from_encoder_deg` ein
   hochaufgelöstes Winkelarray (`n_angle_samples`, 0–360°) auf einem
   gleichmäßigen Zeitgitter berechnet. Keine Spindelposition vorhanden →
   Warnung, leeres Array.
5. **Kraft herunterinterpolieren** (Zeile 1073–1081): Jeder Kraftkanal
   wird geglättet (`MSU.maf`, gleitender Mittelwert mit an die
   Ziel-Samplezahl angepasster Fensterbreite) und dann auf
   `n_force_samples` Punkte linear resampled
   (`MSU.resample_linear_to_n`) — damit Winkel- und Kraftarray punktweise
   zusammenpassen (siehe Konstruktor-Doku zu `n_angle_samples`/
   `n_force_samples`).
6. **Radialkraft-Profil** (Zeile 1084–1092): Liegen sowohl Winkel als
   auch `force_0`/`force_1` (Fx, Fy) vor, wird die Radialkraft `Fr` an der
   Schneidkante berechnet (`MSU.radial_force`) und je Winkel-Bin
   (360 Bins) aufsummiert/gemittelt (`MSU.sum_radial_force_by_angle`).
   Sonst leeres Profil mit Warnung.
7. **Rückgabe** (Zeile 1094–1104): Ein `CutResult`-Objekt mit Rohtabelle,
   Winkelarray, herunterinterpolierten Kraftkanälen, Radialkraftprofil und
   gesammelten Warnungen.

Diese Funktion wird pro erkanntem Schnitt in `run()`,
`export_gap_free_segments()` und `export_z_level_segments()` aufgerufen.

---

## `run(self)` (Zeile 1109–1161)

**Zweck:** Orchestriert die komplette Standard-Pipeline: Block-Erkennung
→ Schnitt-Erkennung je Block → Extraktion je Schnitt → CSV-Export.

**Ablauf:**

1. Falls `load()` noch nicht aufgerufen wurde (keine Force-Dokumente
   geladen), wird es automatisch nachgeholt (Zeile 1116–1117).
2. `blocks = self.detect_blocks()` (Zeile 1119).
3. Für jeden Block (Zeile 1123–1154):
   - `cuts, block_msgs = self.detect_cuts_in_block(block)` —
     Block-Warnungen werden über `warnings.warn` ausgegeben
     (Zeile 1124–1126).
   - Für jeden Schnitt: `extract_cut(...)` aufrufen (Zeile 1129),
     Rohtabelle als CSV `Schnitt_block{XX}_cut{YY}.csv` speichern
     (Zeile 1130–1133), Radialkraftprofil als separates CSV
     `..._radialforce_by_angle.csv` (Zeile 1135–1137), Schnitt-Warnungen
     ausgeben (Zeile 1139–1140).
   - Ergebnis (`CutResult`) wird in `self.results` gesammelt, eine
     Zusammenfassungszeile (`block_id`, `cut_id`, Zeiten, Dauer,
     Zeilenzahl, CSV-Pfad, Warnungsanzahl) in `summary_rows`.
4. `self.summary_ = pd.DataFrame(summary_rows)`, Abschluss-`print` mit
   Anzahl Blöcke/Schnitte (Zeile 1156–1160).

**Rückgabe:** Die Zusammenfassungstabelle (auch als `self.summary_`
zugänglich).

---

## `gap_free_segments(self)` (Zeile 1167–1178)

**Zweck:** Öffentlicher, dünner Wrapper um `_force_bursts()` — liefert
**alle** zusammenhängenden Aufnahmefenster, komplett ungefiltert (auch
kurze Heartbeats), im Gegensatz zu `detect_blocks()`.

**Hintergrund laut Docstring:** Eine Analyse des NC-Programms hat gezeigt,
dass die Aufzeichnung nur *während* eines Schnitts aktiv ist und danach
abgeschaltet wird — jeder Burst entspricht also bereits einer echten
Aufzeichnungssitzung, unabhängig von der (auf Richtungsumkehr basierenden)
Schnitt-Segmentierung in `detect_cuts_in_block()`/`run()`. Das ist eine
**alternative** Segmentierungsstrategie zur bisherigen
block-/richtungsumkehrbasierten.

**Rückgabe:** Liste der Burst-Dicts (`{"t_start", "t_end", "docs"}`),
identisch zu `_force_bursts()`.

---

## `export_gap_free_segments(self, output_dir="./schnitte_gap_free")` (Zeile 1180–1209)

**Zweck:** Exportiert jedes lückenfreie Aufnahmefenster
(`gap_free_segments()`) als eigene CSV — die "einfachere" Alternative zu
`run()`, ohne Dauer-/Amplitudenfilterung und ohne Richtungsumkehr-
Segmentierung: ein Burst = eine Datei.

**Ablauf:**

1. Zielverzeichnis anlegen (Zeile 1190).
2. Für jedes Segment (Zeile 1193–1203): `extract_cut(block_id=0,
   cut_id=seg_id, ...)` aufrufen (Block-ID wird hier nicht verwendet, da
   es keine Block-Struktur gibt — daher `0`), Rohtabelle als
   `gap_free_{seg_id:03d}_t{t0}-{t1}.csv` speichern. **Keine**
   Winkel-/Radialkraft-Zusatzdatei (im Gegensatz zu `run()`).
3. Zusammenfassungstabelle mit `segment_id`, Zeiten, Dauer, Zeilenzahl,
   Pfad, Warnungsanzahl (Zeile 1199–1203).

**Rückgabe:** Zusammenfassungs-`DataFrame`.

---

## `z_level_diagnostics(self, z_axis_index=None, z_tol=0.05, z_safe_heights=(10.0, 20.0))` (Zeile 1217–1308)

**Zweck:** Alternative Gruppierung der Blöcke — nicht zeitlich, sondern
nach **Z-Tiefe** (Standzeitversuch: die Schnitttiefe `ap` wird von Hand
zwischen Versuchen geändert und bleibt für 8 Schnitte/2 Blöcke konstant;
Drehzahl/Vorschub ändern sich alle 4 Schnitte/1 Block).

**Ablauf:**

1. Standardmäßig letzte Vorschubachse (`feed_axis_indices[-1]`, meist
   axis6 = Z) als Z-Achse (Zeile 1260–1262).
2. Für jeden Block aus `detect_blocks()` (Zeile 1264–1287):
   - Machine-Frame holen, exakt maskieren.
   - Fehlen Achsdaten → Zeile mit `NaN`-Werten (Zeile 1271–1277).
   - Sonst: `z_median` (Median statt Mittelwert — **wichtig**, weil kurze
     Eilgang-Transienten bei Z=10mm zwischen Schnitten den Mittelwert
     stark verzerren würden, siehe Docstring Zeile 1230–1234), `z_mad`
     (Median Absolute Deviation), `frac_near_median` (Anteil Samples nahe
     dem Median, Toleranz `z_tol*20`), `near_safe_height` (liegt der
     Median nahe einer bekannten Park-/Sicherheitshöhe wie 10 oder
     20 mm — dann vermutlich kein echter Schnitt).
3. **Level-Bildung** (Zeile 1289–1300): Aufeinanderfolgende Blöcke werden
   zu einem `level_id` zusammengefasst, solange sich `z_median` um
   weniger als `z_tol` ändert; ein Sprung (oder `NaN`) startet ein neues
   Level.
4. `n_blocks_in_level` je Level anhängen (Zeile 1301–1303).

**Rückgabe:** `pd.DataFrame`, eine Zeile je Block, mit Level-Zuordnung und
Diagnosewerten — **keine** Filterung, nur Kennzeichnung (analog zu
`burst_diagnostics()`/`cut_diagnostics()`).

---

## `export_z_level_segments(self, output_dir="./schnitte_z_level", z_axis_index=None, z_tol=0.05)` (Zeile 1310–1363)

**Zweck:** Exportiert je Z-Tiefen-Level (aus `z_level_diagnostics()`)
**eine** zusammengefasste CSV mit den Rohsignalen aller Blöcke dieses
Levels (alle Schnitte gleicher Schnitttiefe aneinandergehängt).

**Ablauf:**

1. Zielverzeichnis anlegen, alte `level_*.csv`-Dateien löschen
   (Zeile 1329–1331) — verhindert "Datei-Leichen" von Levels, die durch
   die Filterung in Schritt 2 in einem früheren Lauf existierten, jetzt
   aber wegfallen.
2. `z_level_diagnostics(...)` aufrufen, dann Blöcke mit `z_median > 0`
   verwerfen (Zeile 1332–1333) — positive Z-Werte sind Eilgang-/
   Park-Höhen (Z=0 ist die Werkstückoberfläche, echte Schnitte haben
   negative Z-Tiefe). Blöcke mit `NaN`-Median (fehlende Achsdaten)
   bleiben erhalten.
3. Für jede Gruppe (`level_id`, Zeile 1337–1356):
   - Für jeden Block der Gruppe: `extract_cut(block_id=..., cut_id=0,
     t_start=block["t_start"], t_end=block["t_end"])` — hier wird der
     **gesamte Block** (nicht einzelne Schnitte) als eine Tabelle
     extrahiert.
   - Alle Block-Tabellen werden per `pd.concat` zusammengehängt und nach
     `time` sortiert (Zeile 1346).
   - Als `level_{level_id:02d}_z{z_depth:.2f}.csv` gespeichert.
   - Zusammenfassungszeile mit `level_id`, `z_depth` (Median der
     Level-Mediane), Anzahl Blöcke, beteiligte `block_ids`,
     `near_safe_height`-Flag, Zeilenzahl, Pfad.
4. Abschluss-`print` (Zeile 1359–1361).

**Rückgabe:** Zusammenfassungs-`DataFrame`, eine Zeile je exportiertem
Level.
