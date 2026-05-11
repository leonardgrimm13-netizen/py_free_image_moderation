# py_free_image_moderation
Ein flexibles Python-Projekt zur **Bild- und GIF-Moderation** mit mehreren Engines (lokal + API), pHash-Listen und klarer CLI-Ausgabe.

**Sprachen:** [English](README.md) | **Deutsch**

## Inhalt
- [Features](#features)
- [Projektstruktur](#projektstruktur)
- [Installation](#installation)
- [Schnellstart](#schnellstart)
- [Verifikation](#verifikation)
- [Wichtige Konfiguration (.env)](#wichtige-konfiguration-env)
- [Ergebnislogik (OK / REVIEW / BLOCK)](#ergebnislogik-ok--review--block)
- [Tipps für den Betrieb](#tipps-für-den-betrieb)

---

## ✨ Features
- **Mehrstufige Moderation** für einzelne Bilder, GIFs, Verzeichnisse und URLs
- **pHash Allowlist/Blocklist** für sehr schnelle Short-Circuit-Entscheidungen
- **OCR-Text-Check** (z. B. gegen Text-Blocklisten)
- Kombinierbare Engines:
  - `OpenNSFW2`
  - `NudeNet`
  - `YOLO` (Waffen-Erkennung)
  - `YOLO forbidden symbols` (Lokale YOLO-Erkennung für verbotene/schädliche Symbole mit `models/forbidden_symbols_yolo.pt`)
  - `OpenAI Moderation` (optional per API-Key)
  - `Sightengine` (optional per API-Credentials)
- **GIF-Handling** mit konfigurierbarem Frame-Sampling
- **JSON-Export** für Weiterverarbeitung in Pipelines
- **Konservative Verdict-Logik** mit nachvollziehbaren Gründen

---

## 📁 Projektstruktur
```text
py_free_image_moderation/
├── moderate_image.py         # Einstiegspunkt (CLI-Wrapper)
├── requirements.txt
├── requirements_api.txt
├── models/
│   └── forbidden_symbols_yolo.pt  # gebündeltes lokales YOLO-Modell für verbotene Symbole
├── data/
│   ├── phash_allowlist.txt
│   ├── phash_blocklist.txt
│   └── ocr_text_blocklist.txt
└── modimg/
    ├── cli.py               # Argumente, Ausgabe, JSON-Export
    ├── pipeline.py          # Ablauf & Engine-Orchestrierung
    ├── verdict.py           # Finale Bewertungslogik
    ├── frames.py            # Bild/GIF-Frame-Laden
    ├── phash.py             # pHash-Utilities
    ├── config.py            # .env-Loading
    └── engines/             # Einzelne Moderations-Engines
```

---

## ⚙️ Installation
> Empfohlen: Python **3.11+** in einer virtuellen Umgebung.

### 1) Repository und venv
```bash
git clone https://github.com/leonardgrimm13-netizen/free_image_moderation_TEST.git
cd free_image_moderation_TEST

python3 -m venv .venv
source .venv/bin/activate
```

Windows-Aktivierung bei Bedarf: `.venv\Scripts\activate`.

### 2) Installationsoptionen

#### A) Offline/Lokal
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Enthält die lokalen Laufzeit- und Engine-Abhängigkeiten (ohne API-Clients):
- `Pillow`
- `numpy`
- `ImageHash`
- `opennsfw2`
- `nudenet`
- `ultralytics`
- `pytesseract`

Damit funktioniert die lokale Pipeline inkl. pHash, lokaler YOLO-Erkennung für verbotene/schädliche Symbole und `--no-apis`.

#### B) Mit APIs
```bash
python3 -m pip install -r requirements_api.txt
```

Enthält alles aus `requirements.txt` plus API-Clients:
- `openai` (OpenAI-Moderation)
- `sightengine` (Sightengine API)

### 3) Dev/Test-Abhängigkeiten
```bash
python3 -m pip install -r requirements_dev.txt
```

Enthält z. B. `pytest` für lokale Testläufe.

### 4) Gebündeltes lokales YOLO-Modell
Dieses Repository enthält `models/forbidden_symbols_yolo.pt` direkt als normale Repository-Datei.

Das Modell wird lokal von der Engine `YOLO forbidden symbols` geladen. Zur Laufzeit werden kein Roboflow und keine externe API verwendet.

### 5) Optionale System-Abhängigkeit für OCR
Für OCR wird in der Regel eine lokale Tesseract-Installation benötigt:
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS (Homebrew): `brew install tesseract`

---

## 🚀 Schnellstart

### Einzelnes Bild prüfen
```bash
python3 moderate_image.py /pfad/zum/bild.jpg
```

### GIF prüfen (Frame-Sampling)
```bash
python3 moderate_image.py /pfad/zur/datei.gif --sample-frames 12
```

### URL prüfen
```bash
python3 moderate_image.py "https://example.com/image.jpg"
```

### Verzeichnis prüfen
```bash
python3 moderate_image.py ./images --recursive
```

### Ohne externe APIs (Basisinstallation ausreichend)
```bash
python3 moderate_image.py ./images --recursive --no-apis
```

Die lokale YOLO-Symbolerkennung erscheint mit kompakten numerischen Scores, zum Beispiel:
```text
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.72, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.93, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=1.00
```

### JSON-Report schreiben
```bash
python3 moderate_image.py ./images --recursive --json moderation_report.json
```

### Benchmark-Modus
Der Benchmark-Modus misst Laufzeiten pro Datei und pro Engine, ohne Moderationsentscheidungen zu verändern.

```bash
python3 moderate_image.py ./images --recursive --no-apis --benchmark
python3 moderate_image.py ./images --recursive --no-apis --benchmark-json benchmark.json
python3 moderate_image.py ./images --recursive --no-apis --json moderation_report.json --benchmark-json benchmark.json
```

Das Benchmark-JSON-Feld `total_wall_ms` enthält nur die Wall-Clock-Zeit für die Verarbeitung der Eingaben (nicht die Zeit für das Schreiben von JSON-Ausgabedateien).

**Exit Codes:**
- `0` = alle Ergebnisse `OK`
- `2` = mindestens ein Ergebnis nicht `OK`

---

## ✅ Verifikation
```bash
python3 -m compileall -q .
pytest -q
python3 moderate_image.py --help
python3 moderate_image.py pfad/zum/test.png --no-apis
```

Erwartetes Verhalten (kurz):
- `python3 -m compileall -q .` → Exitcode `0` bei syntaktisch gültigem Code.
- `pytest -q` → Exitcode `0` bei erfolgreichen Tests, sonst ungleich `0`.
- `python3 moderate_image.py --help` → Exitcode `0` und Anzeige der CLI-Hilfe.
- `python3 moderate_image.py pfad/zum/test.png --no-apis` → Exitcode `0`, wenn die Eingabe `OK` ist, oder `2`, wenn sie `REVIEW`/`BLOCK` ergibt.

Optionale Engines dürfen fehlen; sie müssen in der Ausgabe sauber als `skipped`/`disabled` erscheinen, statt die Ausführung abzubrechen.

---

## 🔧 Wichtige Konfiguration (.env)
Das Projekt lädt automatisch `.env` aus dem Projekt-Root. Beispiel:

Der Loader prüft `.env`, dann `.env.txt` und nutzt `.env.example` als Fallback-Defaults. Für beste Ergebnisse `.env.example` nach `.env` kopieren und die `.env` anpassen.

```env
# API-Engines
OPENAI_API_KEY=...
SIGHTENGINE_USER=...
SIGHTENGINE_SECRET=...

# Global
SAMPLE_FRAMES=12
SHORT_CIRCUIT_PHASH=1
ENGINE_ERROR_POLICY=review

# OCR
OCR_ENABLE=1
OCR_LANG=eng

# pHash Auto-Learn
PHASH_AUTO_LEARN_ENABLE=0
PHASH_AUTO_ALLOW_APPEND=0
PHASH_AUTO_BLOCK_APPEND=0
# pHash-Auto-Lernen ist standardmäßig aus; erst nach Prüfung von Thresholds und False Positives aktivieren.

# Lokales YOLO-Modell für verbotene/schädliche Symbole
FORBIDDEN_SYMBOLS_YOLO_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_MODEL=models/forbidden_symbols_yolo.pt
FORBIDDEN_SYMBOLS_YOLO_CONF=0.20
FORBIDDEN_SYMBOLS_YOLO_IOU=0.45
FORBIDDEN_SYMBOLS_YOLO_IMGSZ=960
FORBIDDEN_SYMBOLS_YOLO_MAX_DET=20
FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=2
FORBIDDEN_SYMBOLS_YOLO_DEVICE=auto
FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30
FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90
FORBIDDEN_SYMBOLS_YOLO_INCLUDE_BOXES=1
FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=
```

Nützliche Schalter:
- Wichtige Performance-Regler: `SAMPLE_FRAMES`, `API_POLICY`, `YOLO_IMGSZ`, `YOLO_MAX_FRAMES`, `YOLO_MAX_DET`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES`, `OCR_MAX_FRAMES`, `PHASH_ALLOW_MAX_DISTANCE`, `PHASH_BLOCK_MAX_DISTANCE`
- `API_POLICY=always|on_review|never` steuert, wann API-Engines laufen
- `OPENAI_DISABLE=1` / `SIGHTENGINE_*` weglassen, wenn API-Engines nicht genutzt werden
- `PHASH_ALLOW_DISABLE=1` oder `PHASH_BLOCK_DISABLE=1` zum gezielten Abschalten
- `SCORE_VERBOSE=1` für ausführlichere Engine-Scores
- `MODIMG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` für die zentrale Protokollierung
- `MODIMG_PARALLEL_ENGINES=1` unabhängige Engines gleichzeitig ausführen (optional/experimentell; standardmäßig deaktiviert)
- `NO_CHECKS_POLICY=review` steuert den Fallback, wenn keine Engine lief: `ok` = erlauben, `review` = sicherer Standard, `block` = strengster Modus


### Lokale YOLO-Konfiguration für verbotene Symbole
- `FORBIDDEN_SYMBOLS_YOLO_ENABLE=1` aktiviert standardmäßig das gebündelte lokale Modell.
- `FORBIDDEN_SYMBOLS_YOLO_CONF=0.20` steuert die rohe YOLO-Erkennungs-Confidence.
- `FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30` steuert, ab wann Funde das Urteil auf `REVIEW` anheben.
- `FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90` steuert, ab wann Funde das Urteil auf `BLOCK` anheben.
- Empfohlene Defaults: `conf=0.20`, `review=0.30`, `block=0.90`, `imgsz=960`.
- Für schnellere CPU-Scans: `FORBIDDEN_SYMBOLS_YOLO_IMGSZ=640`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=1` und `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`.
- Unzuverlässige Klassen können zur Laufzeit ignoriert werden, z. B. `FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=communism,antifa`.

---

## 🧠 Ergebnislogik (OK / REVIEW / BLOCK)
- **Staged Pipeline:** `pHash` → lokale Engines → optionale API-Engines → finales Urteil
- **pHash-Short-Circuit** kann früh entscheiden:
  - Allowlist-Treffer → direkt `OK`
  - Blocklist-Treffer → direkt `BLOCK`
- Wenn pHash nicht per Short-Circuit entscheidet, laufen die lokalen Engines weiter, inklusive `YOLO forbidden symbols`.
- Die YOLO-Engine für verbotene Symbole trägt zum Hate/Policy-Risiko bei: ab dem Block-Schwellenwert sollte das Ergebnis `BLOCK` werden, ab dem Review-Schwellenwert `REVIEW`.
- Erkennungslabels und Boxen werden im JSON unter `details.detections` gespeichert.
- `verdict.py` verdichtet Signale (Nudity, Violence, Hate) zu finalem Urteil
- Fehlerverhalten lässt sich über `ENGINE_ERROR_POLICY` steuern (`ignore`, `review`, `block`)

---

## 🛠️ Tipps für den Betrieb
- Starte zuerst mit `--no-apis`, um lokale Pipeline und Performance zu prüfen.
- Nutze `--json`, wenn Ergebnisse in CI/CD oder Backend-Services weiterverarbeitet werden sollen.
- Pflege `data/phash_allowlist.txt` und `data/phash_blocklist.txt` regelmäßig für stabile Entscheidungen bei wiederkehrendem Content.
- Bei GIFs ggf. `--sample-frames` erhöhen, wenn problematischer Content nur in einzelnen Frames auftaucht.
