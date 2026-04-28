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
git clone https://github.com/leonardgrimm13-netizen/py_free_image_moderation.git
cd py_free_image_moderation

python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 2) Installationsoptionen

#### A) Offline/Lokal
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Enthält die lokalen Laufzeit- und Engine-Abhängigkeiten (ohne API-Clients):
- `Pillow`
- `numpy`
- `ImageHash`
- `opennsfw2`
- `nudenet`
- `ultralytics`
- `pytesseract`

Damit funktioniert die lokale Pipeline inkl. pHash und `--no-apis`.

#### B) Mit APIs
```bash
pip install -r requirements_api.txt
```

Enthält alles aus `requirements.txt` plus API-Clients:
- `openai` (OpenAI-Moderation)
- `sightengine` (Sightengine API)

### 3) Dev/Test-Abhängigkeiten
```bash
pip install -r requirements_dev.txt
```

Enthält z. B. `pytest` für lokale Testläufe.

### 4) Optionale System-Abhängigkeit für OCR
Für OCR wird in der Regel eine lokale Tesseract-Installation benötigt:
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS (Homebrew): `brew install tesseract`

---

## 🚀 Schnellstart

### Einzelnes Bild prüfen
```bash
python moderate_image.py /pfad/zum/bild.jpg
```

### GIF prüfen (Frame-Sampling)
```bash
python moderate_image.py /pfad/zur/datei.gif --sample-frames 12
```

### URL prüfen
```bash
python moderate_image.py "https://example.com/image.jpg"
```

### Verzeichnis prüfen
```bash
python moderate_image.py ./images --recursive
```

### Ohne externe APIs (Basisinstallation ausreichend)
```bash
python moderate_image.py ./images --recursive --no-apis
```

### JSON-Report schreiben
```bash
python moderate_image.py ./images --recursive --json moderation_report.json
```

**Exit Codes:**
- `0` = alle Ergebnisse `OK`
- `2` = mindestens ein Ergebnis nicht `OK`

---

## ✅ Verifikation
```bash
python -m compileall -q .
pytest -q
python moderate_image.py --help
python moderate_image.py pfad/zum/test.png --no-apis
```

Erwartetes Verhalten (kurz):
- `python -m compileall -q .` → Exitcode `0` bei syntaktisch gültigem Code.
- `pytest -q` → Exitcode `0` bei erfolgreichen Tests, sonst ungleich `0`.
- `python moderate_image.py --help` → Exitcode `0` und Anzeige der CLI-Hilfe.
- `python moderate_image.py pfad/zum/test.png --no-apis` → Exitcode `0`, wenn die Eingabe `OK` ist, oder `2`, wenn sie `REVIEW`/`BLOCK` ergibt.

Optionale Engines dürfen fehlen; sie müssen in der Ausgabe sauber als `skipped`/`disabled` erscheinen, statt die Ausführung abzubrechen.

---

## 🔧 Wichtige Konfiguration (.env)
Das Projekt lädt automatisch `.env` aus dem Projekt-Root. Beispiel:

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
PHASH_AUTO_ALLOW_APPEND=1
PHASH_AUTO_BLOCK_APPEND=1
```

Nützliche Schalter:
- `OPENAI_DISABLE=1` / `SIGHTENGINE_*` weglassen, wenn API-Engines nicht genutzt werden
- `PHASH_ALLOW_DISABLE=1` oder `PHASH_BLOCK_DISABLE=1` zum gezielten Abschalten
- `SCORE_VERBOSE=1` für ausführlichere Engine-Scores
- `MODIMG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` für die zentrale Protokollierung
- `MODIMG_PARALLEL_ENGINES=1` unabhängige Engines gleichzeitig ausführen (optional/experimentell; standardmäßig deaktiviert)
- `NO_CHECKS_POLICY=review` steuert den Fallback, wenn keine Engine lief: `ok` = erlauben, `review` = sicherer Standard, `block` = strengster Modus

---

## 🧠 Ergebnislogik (OK / REVIEW / BLOCK)
- **pHash-Short-Circuit** kann früh entscheiden:
  - Allowlist-Treffer → direkt `OK`
  - Blocklist-Treffer → direkt `BLOCK`
- Danach werden die restlichen Engines aggregiert
- `verdict.py` verdichtet Signale (Nudity, Violence, Hate) zu finalem Urteil
- Fehlerverhalten lässt sich über `ENGINE_ERROR_POLICY` steuern (`ignore`, `review`, `block`)

---

## 🛠️ Tipps für den Betrieb
- Starte zuerst mit `--no-apis`, um lokale Pipeline und Performance zu prüfen.
- Nutze `--json`, wenn Ergebnisse in CI/CD oder Backend-Services weiterverarbeitet werden sollen.
- Pflege `data/phash_allowlist.txt` und `data/phash_blocklist.txt` regelmäßig für stabile Entscheidungen bei wiederkehrendem Content.
- Bei GIFs ggf. `--sample-frames` erhöhen, wenn problematischer Content nur in einzelnen Frames auftaucht.
