[README.de.md](https://github.com/user-attachments/files/30100764/README.de.md)
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
- [Tipps für den Betrieb](#tipps-fuer-den-betrieb)

---

<a name="features"></a>
## ✨ Features
- **Mehrstufige Moderation** für einzelne Bilder, GIFs, Verzeichnisse und URLs
- **pHash Allowlist/Blocklist** für sehr schnelle Short-Circuit-Entscheidungen
- **OCR-Text-Check** (z. B. gegen Text-Blocklisten)
- Kombinierbare Engines:
  - `OpenNSFW2`
  - `NudeNet`
  - `YOLO Waffen-Erkennung` (lokale YOLO-Erkennung mit `models/yolov8s-oiv7.pt`)
  - `YOLO forbidden symbols` (Lokale YOLO-Erkennung für verbotene/schädliche Symbole mit `models/forbidden_symbols_yolo.pt`)
  - `OpenAI Moderation` (optional per API-Key)
  - `Sightengine` (optional per API-Credentials)
- **GIF-Handling** mit konfigurierbarem Frame-Sampling
- **JSON-Export** für Weiterverarbeitung in Pipelines
- **Konservative Verdict-Logik** mit nachvollziehbaren Gründen

---

<a name="projektstruktur"></a>
## 📁 Projektstruktur
```text
py_free_image_moderation/
├── moderate_image.py         # Einstiegspunkt (CLI-Wrapper)
├── requirements.txt        # Core-Runtime
├── requirements_local.txt  # lokale Vision/OCR-Engines
├── requirements_api.txt    # API-Engines
├── requirements_all.txt    # lokale + API-Runtime
├── requirements_dev.txt    # Tests/Lint/Build-Tools
├── models/
│   ├── forbidden_symbols_yolo.pt  # gebündeltes lokales YOLO-Modell für verbotene Symbole
│   └── yolov8s-oiv7.pt            # gebündeltes lokales YOLO-Modell für Waffen-Erkennung
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

<a name="installation"></a>
## ⚙️ Installation
> Empfohlen und für dieses Projekt unterstützt: Python **3.11 oder 3.12** in einer virtuellen Umgebung.
>
> `pyproject.toml` deklariert `>=3.11,<3.13`. Python 3.13+ wird nicht als unterstützt versprochen, solange du es nicht selbst testest.

### 1) Repository und venv
Linux/macOS:
```bash
git clone https://github.com/leonardgrimm13-netizen/py_free_image_moderation.git
cd py_free_image_moderation

python3.12 -m venv .venv  # oder: python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:
```powershell
git clone https://github.com/leonardgrimm13-netizen/py_free_image_moderation.git
cd py_free_image_moderation

py -3.12 -m venv .venv  # oder: py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Installationsoptionen

#### A) Basis/Core
```bash
python -m pip install -r requirements.txt
```

Enthält nur die leichten Core-Abhängigkeiten:
- `Pillow`
- `numpy`
- `ImageHash`

Das reicht für Bildladen, GIF-Frame-Sampling, pHash-Allow/Blocklisten, JSON-Ausgabe und sauberes Überspringen optionaler Engines.

#### B) Lokal/Vision
```bash
python -m pip install -r requirements_local.txt
```

Enthält die Basis-Abhängigkeiten plus lokale Vision/OCR-Engines:
- `opennsfw2[tf-keras]`
- `nudenet`
- `ultralytics`
- `pytesseract`

Damit funktioniert die lokale Pipeline inkl. OpenNSFW2, NudeNet, YOLO-Waffen, lokaler YOLO-Erkennung für verbotene/schädliche Symbole, OCR-Python-Bindings und `--no-apis`.

OpenNSFW2 wird absichtlich mit dem Extra `tf-keras` installiert. OpenNSFW2 braucht ein Backend; dieses Projekt nutzt den TensorFlow/tf-keras-Pfad als stabilen Standard für Python 3.11/3.12. Das reine Paket `opennsfw2` reicht für zuverlässige lokale Inferenz nicht aus.

#### C) Mit APIs
```bash
python -m pip install -r requirements_api.txt
```

Enthält die Basis-Abhängigkeiten plus API-Clients:
- `openai` (OpenAI-Moderation)
- `requests` (HTTP-Client für die Sightengine-Engine)

Der Code ruft Sightengine direkt per HTTP auf; ein separates `sightengine`-SDK wird nicht importiert.

#### D) Alle Runtime-Engines
```bash
python -m pip install -r requirements_all.txt
```

Editable-Installationen nutzen dieselbe Trennung über Extras:
```bash
python -m pip install -e ".[dev]"      # Tests/Linting
python -m pip install -e ".[local]"    # lokale Vision-Engines
python -m pip install -e ".[api]"      # API-Engines
python -m pip install -e ".[all]"      # lokale Vision- und API-Engines
```

### 3) Dev/Test-Abhängigkeiten
```bash
python -m pip install -r requirements_dev.txt
```

Enthält die Basis-Abhängigkeiten plus `pytest`, `pytest-cov`, `ruff` und `build`.

### 4) Gebündelte lokale YOLO-Modelle
Dieses Repository enthält `models/forbidden_symbols_yolo.pt` und `models/yolov8s-oiv7.pt` direkt als normale Repository-Dateien.

Das Modell wird lokal von der Engine `YOLO forbidden symbols` geladen. Zur Laufzeit werden kein Roboflow und keine externe API verwendet. Falls die Datei fehlt, setze `FORBIDDEN_SYMBOLS_YOLO_MODEL` auf einen absoluten Pfad oder starte aus dem Projekt-Root. Falls Git-LFS nur eine Pointer-Datei statt echter Gewichte ausgecheckt hat, führe `git lfs pull` aus.

Die separate Engine `YOLO-World weapons` verwendet standardmäßig das gebündelte `models/yolov8s-oiv7.pt`. Sowohl Source-Checkouts als auch installierte Wheels lösen gebündelte Ressourcen automatisch auf. Für eigene Waffen-Gewichte setze `YOLO_WEAPON_MODEL=/absoluter/oder/projektrelativer/pfad.pt` oder `YOLO_WORLD_MODEL=/absoluter/oder/projektrelativer/pfad.pt`.

### 5) Optionale System-Abhängigkeit für OCR
Für OCR wird in der Regel eine lokale Tesseract-Installation benötigt:
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS (Homebrew): `brew install tesseract`

---

<a name="schnellstart"></a>
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

### Mehrere explizite Eingaben prüfen
```bash
python moderate_image.py erstes.jpg zweites.png animation.gif --no-apis
```

Wiederholte oder überlappende lokale Pfade werden einmal verarbeitet; die Reihenfolge des ersten Auftretens bleibt erhalten.

### Verzeichnis mit Datei-Parallelisierung prüfen
```bash
python moderate_image.py ./images --recursive --file-workers 2
```

### Ohne externe APIs
```bash
python moderate_image.py ./images --recursive --no-apis
```

Mit nur der Basisinstallation melden optionale lokale Vision-Engines `skipped`, statt abzustürzen. Installiere `requirements_local.txt` oder `.[local]`, wenn lokale Inferenz laufen soll.

Die lokale YOLO-Symbolerkennung erscheint mit kompakten numerischen Scores, zum Beispiel:
```text
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.72, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=0.00
[ok] YOLO forbidden symbols (...) forbidden_symbols_max_conf=0.93, forbidden_symbols_review_hit=1.00, forbidden_symbols_block_hit=1.00
```

### JSON-Report schreiben
```bash
python moderate_image.py ./images --recursive --json moderation_report.json
```

### Benchmark-Modus
Der Benchmark-Modus misst Laufzeiten pro Datei und pro Engine, ohne Moderationsentscheidungen zu verändern.

```bash
python moderate_image.py ./images --recursive --no-apis --benchmark
python moderate_image.py ./images --recursive --no-apis --benchmark-json benchmark.json
python moderate_image.py ./images --recursive --no-apis --json moderation_report.json --benchmark-json benchmark.json
```

Das Benchmark-JSON-Feld `total_wall_ms` enthält nur die Wall-Clock-Zeit für die Verarbeitung der Eingaben (nicht die Zeit für das Schreiben von JSON-Ausgabedateien).

**Exit Codes:**
- `0` = alle Ergebnisse `OK`
- `2` = mindestens ein Ergebnis nicht `OK`

---

<a name="verifikation"></a>
## ✅ Verifikation
Core-Installation:
```bash
python -m pip install -r requirements.txt
python -m compileall -q .
python moderate_image.py --help
python moderate_image.py pfad/zum/test.png --no-apis
python -m pip check
```

Dev/Test-Installation:
```bash
python -m pip install -r requirements_dev.txt
pytest -q
```

Local/Vision-Smoke-Test:
```bash
python -m pip install -r requirements_local.txt
python -c "import opennsfw2, nudenet, ultralytics, pytesseract"
python -m pip check
```

Erwartetes Verhalten (kurz):
- `python -m compileall -q .` → Exitcode `0` bei syntaktisch gültigem Code.
- `pytest -q` → Exitcode `0` bei erfolgreichen Tests, sonst ungleich `0`.
- `python moderate_image.py --help` → Exitcode `0` und Anzeige der CLI-Hilfe.
- `python moderate_image.py pfad/zum/test.png --no-apis` → Exitcode `0`, wenn die Eingabe `OK` ist, oder `2`, wenn sie `REVIEW`/`BLOCK` ergibt.

Optionale Engines dürfen fehlen; sie müssen in der Ausgabe sauber als `skipped`/`disabled` erscheinen, statt die Ausführung abzubrechen.

---

<a name="wichtige-konfiguration-env"></a>
## 🔧 Wichtige Konfiguration (.env)
Bibliotheksimporte laden `.env` automatisch aus dem Quell-/Paket-Root. Die installierte CLI prüft zusätzlich zuerst das aktuelle Arbeitsverzeichnis. Dadurch kann sie eine projektlokale `.env` verwenden, ohne dass normale Importe beliebigen Arbeitsverzeichnissen vertrauen. Beispiel:

Der Loader prüft an jedem Ort `.env`, dann `.env.txt`. `.env.example` wird nicht automatisch geladen, weil diese Datei Dokumentation ist und Platzhalter-Credentials oder schwere optionale Engine-Einstellungen enthalten kann. Für beste Ergebnisse `.env.example` nach `.env` kopieren und die `.env` anpassen.

Ohne `OPENAI_CACHE_PATH` verwendet der OpenAI-Cache unter Linux `XDG_CACHE_HOME` beziehungsweise `~/.cache`, unter Windows `LOCALAPPDATA` und unter macOS `~/Library/Caches`, jeweils im Unterordner `py-free-image-moderation`. Ein expliziter absoluter Pfad wird unverändert verwendet. Aus Rückwärtskompatibilitätsgründen bleibt ein expliziter relativer `OPENAI_CACHE_PATH` relativ zum Quell-/Paket-Root.

```env
# API-Engines
OPENAI_API_KEY=...
SIGHTENGINE_USER=...
SIGHTENGINE_SECRET=...

# Global
SAMPLE_FRAMES=12
SHORT_CIRCUIT_PHASH=1
ENGINE_ERROR_POLICY=review
API_POLICY=always
MODIMG_PARALLEL_ENGINES=0
MODIMG_PARALLEL_WORKERS=4
MODIMG_FILE_WORKERS=1
MODIMG_MAX_FILE_WORKERS=32

# Limits für nicht vertrauenswürdige Eingaben
MODIMG_MAX_DOWNLOAD_BYTES=25000000
MODIMG_URL_TIMEOUT_SEC=20
MODIMG_MAX_URL_REDIRECTS=5
MODIMG_ALLOW_PRIVATE_URLS=0
MODIMG_MAX_IMAGE_PIXELS=64000000
MODIMG_MAX_ANIMATION_FRAMES=5000
MODIMG_MAX_DECODED_PIXELS=256000000

# OCR
OCR_ENABLE=1
OCR_LANG=eng

# pHash Auto-Learn
PHASH_AUTO_LEARN_ENABLE=0
PHASH_AUTO_ALLOW_APPEND=0
PHASH_AUTO_BLOCK_APPEND=0
# pHash-Auto-Lernen ist standardmäßig aus; erst nach Prüfung von Thresholds und False Positives aktivieren.

# Optionales YOLO-World-Waffenmodell
# Wenn leer und models/yolov8s-oiv7.pt fehlt, wird die Engine übersprungen.
YOLO_WEAPON_MODEL=
YOLO_WORLD_MODEL=
YOLO_CONF=0.25
YOLO_IMGSZ=640
YOLO_MAX_FRAMES=2
YOLO_DEVICE=
YOLO_BATCH_ENABLE=1

# OpenNSFW2 Speed/Stabilitaet
OPENNSFW2_IN_PROCESS=0

# Lokales YOLO-Modell für verbotene/schädliche Symbole
FORBIDDEN_SYMBOLS_YOLO_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_MODEL=models/forbidden_symbols_yolo.pt
FORBIDDEN_SYMBOLS_YOLO_CONF=0.20
FORBIDDEN_SYMBOLS_YOLO_IOU=0.45
FORBIDDEN_SYMBOLS_YOLO_IMGSZ=960
FORBIDDEN_SYMBOLS_YOLO_MAX_DET=20
FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=2
FORBIDDEN_SYMBOLS_YOLO_DEVICE=auto
FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK=1
FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30
FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90
FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF=
FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF=
FORBIDDEN_SYMBOLS_YOLO_INCLUDE_BOXES=1
FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=
```

Nützliche Schalter:
- Wichtige Performance-Regler: `SAMPLE_FRAMES`, `API_POLICY`, `MODIMG_FILE_WORKERS`, `MODIMG_PARALLEL_ENGINES`, `MODIMG_PARALLEL_WORKERS`, `OPENNSFW2_IN_PROCESS`, `YOLO_BATCH_ENABLE`, `YOLO_IMGSZ`, `YOLO_MAX_FRAMES`, `YOLO_MAX_DET`, `FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ`, `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES`, `OCR_MAX_FRAMES`, `PHASH_ALLOW_MAX_DISTANCE`, `PHASH_BLOCK_MAX_DISTANCE`
- `API_POLICY=always|on_review|never` steuert, wann API-Engines laufen
- `OPENAI_DISABLE=1` / `SIGHTENGINE_*` weglassen, wenn API-Engines nicht genutzt werden
- `PHASH_ALLOW_DISABLE=1` oder `PHASH_BLOCK_DISABLE=1` zum gezielten Abschalten
- `SCORE_VERBOSE=1` für ausführlichere Engine-Scores
- `MODIMG_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR` für die zentrale Protokollierung
- `MODIMG_PARALLEL_ENGINES=1` unabhängige Engines gleichzeitig ausführen (optional/experimentell; standardmäßig deaktiviert)
- `MODIMG_FILE_WORKERS=2` oder `--file-workers 2` verarbeitet mehrere Input-Dateien parallel; Standard bleibt `1`.
- `OPENNSFW2_IN_PROCESS=0` ist der robuste Default und nutzt einen isolierten Subprozess. Setze `OPENNSFW2_IN_PROCESS=1` nur für schnellere In-Process-Prediction; native TensorFlow-Abstürze können dann den gesamten CLI-Prozess beenden. `auto` versucht zuerst In-Process und fällt bei normalen Python-Fehlern einmal auf den Subprozess zurück.
- `NO_CHECKS_POLICY=review` steuert den Fallback, wenn keine Engine lief: `ok` = erlauben, `review` = sicherer Standard, `block` = strengster Modus
- `YOLO_WEAPON_MODEL` oder `YOLO_WORLD_MODEL` verweist auf eigene YOLO-Waffen-Gewichte; ohne Gewichte wird die Waffen-Engine übersprungen, nicht als Fehler gewertet.


### Fast-Preset
Empfehlung für den ersten Performance-Test: `MODIMG_FILE_WORKERS=2` und `MODIMG_PARALLEL_WORKERS=4`.

```env
MODIMG_PARALLEL_ENGINES=1
MODIMG_PARALLEL_WORKERS=4
MODIMG_FILE_WORKERS=2
OPENNSFW2_IN_PROCESS=1
YOLO_BATCH_ENABLE=1
FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE=1
YOLO_MAX_FRAMES=1
FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES=1
SAMPLE_FRAMES=4
API_POLICY=on_review
```

Speed-Tradeoffs:
- Weniger geprüfte Frames können die Genauigkeit bei GIFs/animierten Bildern reduzieren, wenn relevante Inhalte nur in übersprungenen Frames vorkommen.
- Zu viele File-Worker können GPU/VRAM überlasten, besonders wenn beide YOLO-Engines auf der GPU laufen.
- Starte mit File-Workers `2` und Engine-Workers `4`, dann per Benchmark erhöhen.

### Lokale YOLO-Konfiguration für verbotene Symbole
- `FORBIDDEN_SYMBOLS_YOLO_ENABLE=1` aktiviert standardmäßig das gebündelte lokale Modell.
- `FORBIDDEN_SYMBOLS_YOLO_CONF=0.20` steuert die rohe YOLO-Erkennungs-Confidence.
- `FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF=0.30` steuert, ab wann Funde das Urteil auf `REVIEW` anheben.
- `FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF=0.90` steuert, ab wann Funde das Urteil auf `BLOCK` anheben.
- `FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF` und `FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF` überschreiben Schwellen pro Label, z. B. `swastika:0.50,isis:0.75`.
- Empfohlene Defaults: `conf=0.20`, `review=0.30`, `block=0.90`, `imgsz=960`.
- `FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0` deaktiviert die Frame-Inferenz dieser Engine und liefert `skipped`; dies zählt nicht als erfolgreicher Moderationscheck.
- Für schnellere CPU-only-Scans: Fast-Preset plus `OCR_MAX_FRAMES=1`, `YOLO_IMGSZ=416`, `YOLO_DEVICE=cpu`, `FORBIDDEN_SYMBOLS_YOLO_IMGSZ=640` und `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu`.
- Unzuverlässige Klassen können zur Laufzeit ignoriert werden, z. B. `FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS=communism,antifa`.

---

<a name="ergebnislogik-ok--review--block"></a>
## 🧠 Ergebnislogik (OK / REVIEW / BLOCK)
- **Staged Pipeline:** `pHash` → lokale Engines → optionale API-Engines → finales Urteil
- **pHash-Short-Circuit** kann früh entscheiden:
  - Allowlist-Treffer → direkt `OK`
  - Blocklist-Treffer → direkt `BLOCK`
- Wenn pHash nicht per Short-Circuit entscheidet, laufen die lokalen Engines weiter, inklusive `YOLO forbidden symbols`.
- Mit `SHORT_CIRCUIT_PHASH=0` wird ein Allowlist-Treffer nur als Kontext erfasst und kann einen späteren Blocklist-, OCR-, Local-Engine- oder API-Block nicht überschreiben.
- Die YOLO-Engine für verbotene Symbole trägt zum Hate/Policy-Risiko bei: ab dem Block-Schwellenwert sollte das Ergebnis `BLOCK` werden, ab dem Review-Schwellenwert `REVIEW`.
- Erkennungslabels und Boxen werden im JSON unter `details.detections` gespeichert.
- `verdict.py` verdichtet Signale (Nudity, Violence, Hate) zu finalem Urteil
- Fehlerverhalten lässt sich über `ENGINE_ERROR_POLICY` steuern (`ignore`, `review`, `block`)

### Sicherheitslimits für nicht vertrauenswürdige Eingaben
- URL-Eingaben sind auf HTTP(S) beschränkt, verbieten eingebettete Zugangsdaten, validieren alle DNS-Antworten, binden die validierte Adresse an die Verbindung und prüfen jedes Redirect-Ziel erneut. Loopback-, private, link-lokale, Multicast- und sonstige nicht öffentliche Ziele werden standardmäßig abgelehnt.
- `MODIMG_ALLOW_PRIVATE_URLS=1` deaktiviert den SSRF-Schutz für private Netze. Diese Option nur für vertrauenswürdige URLs in einem kontrollierten Netz verwenden.
- URL-Downloads werden gestreamt und haben Gesamt-Timeout, Redirect-Limit, Byte-Limit, Formatsignaturprüfung sowie garantierte Bereinigung partieller temporärer Dateien. Query-Werte und URL-Zugangsdaten werden aus Reports und Fehlern entfernt.
- Pillow-Decode-Limits begrenzen Dimensionen, Pixel, Animationsframes, gesampelte Frames und die Summe dekodierter Pixel. Ein ausgewählter, nicht dekodierbarer Animationsframe lässt den Loader fehlschlagen, statt still übersprungen zu werden.
- pHash-Listen und OpenAI-Cache nutzen begrenzte Lesevorgänge und atomaren Austausch. Prozessinterne Locks serialisieren Cache-/Listen-Schreibzugriffe und Inferenz auf gecachten Modellen.
- Gebündelte Modell-Defaults werden ausschließlich aus Quellpaket- und Installationspfaden aufgelöst; das aktuelle Arbeitsverzeichnis ist niemals ein automatischer Fallback. Explizit konfigurierte relative Ressourcenpfade werden zuerst aus dem Arbeitsverzeichnis und danach über Quellpaket- und Installations-Fallbacks aufgelöst. Jede eigene `.pt`-Datei ist als vertrauenswürdige ausführbare Modelldatei zu behandeln.
- OCR-Blocklisten behandeln normale Zeilen als Literaltext. Bewusste reguläre Ausdrücke erhalten das Präfix `re:`.

---

<a name="tipps-fuer-den-betrieb"></a>
## 🛠️ Tipps für den Betrieb
- Starte zuerst mit `--no-apis`, um lokale Pipeline und Performance zu prüfen.
- Nutze `--json`, wenn Ergebnisse in CI/CD oder Backend-Services weiterverarbeitet werden sollen.
- Pflege `data/phash_allowlist.txt` und `data/phash_blocklist.txt` regelmäßig für stabile Entscheidungen bei wiederkehrendem Content.
- Bei GIFs ggf. `--sample-frames` erhöhen, wenn problematischer Content nur in einzelnen Frames auftaucht.

## Troubleshooting
- `pip install -r requirements.txt` installiert nichts oder verhält sich seltsam: prüfe, dass jede Dependency auf einer eigenen Zeile steht. Eine beschädigte Requirements-Datei mit allen Dependencies in einer Zeile oder versehentlich auskommentierten Dependency-Zeilen ist ungültig.
- Nicht unterstützte Python-Version: verwende Python 3.11 oder 3.12. Python 3.13+ kann für einzelne Pakete funktionieren, wird hier aber nicht versprochen und der ML-Stack kann es ablehnen.
- OpenNSFW2-Backend fehlt: installiere `requirements_local.txt` oder `.[local]`. Das Projekt nutzt absichtlich `opennsfw2[tf-keras]`; reines `opennsfw2` kann importierbar sein, ohne ein nutzbares Inferenz-Backend zu haben.
- Nativer OpenNSFW2-Absturz: der Default ist der isolierte Python-Subprozess (`OPENNSFW2_IN_PROCESS=0`), damit TensorFlow/native Abstürze zu einem kontrollierten Engine-Fehler werden, statt die CLI zu beenden. `OPENNSFW2_IN_PROCESS=1` ist schneller, aber weniger robust, weil native TensorFlow-Abstürze den gesamten Prozess beenden können. `OPENNSFW2_IN_PROCESS=auto` versucht zuerst In-Process und fällt bei normalen Python-Exceptions einmal auf den Subprozess zurück, kann aber native Prozessabstürze nicht abfangen.
- TensorFlow/Keras-Kompatibilität: nutze eine frische Python-3.11/3.12-venv. Wenn du Keras/TensorFlow-Pakete vorher manuell installiert hast, erstelle die venv neu und installiere `requirements_local.txt` erneut.
- Tesseract fehlt: installiere die System-Binary und setze `TESSERACT_CMD`, falls sie nicht im `PATH` liegt. Die OCR-Engine liefert `skipped`/kontrollierte `error` statt abzustürzen.
- CUDA/GPU nicht verfügbar: die CLI setzt aus Prozesssicherheitsgründen standardmäßig `CUDA_VISIBLE_DEVICES=-1`. Für CPU-only-Läufe kannst du zusätzlich `YOLO_DEVICE=cpu` und `FORBIDDEN_SYMBOLS_YOLO_DEVICE=cpu` setzen; für GPU-Läufe setze `CUDA_VISIBLE_DEVICES` explizit vor dem CLI-Start.
- YOLO-Modell fehlt: setze `FORBIDDEN_SYMBOLS_YOLO_MODEL` oder `YOLO_WEAPON_MODEL` auf einen absoluten Pfad. Fehlende optionale Gewichte werden als `skipped` gemeldet.
- Git-LFS-Pointer statt echter `.pt`: führe `git lfs pull` aus; Pointer-Dateien werden erkannt und mit klarer Meldung übersprungen.
- API-Keys fehlen: `OPENAI_API_KEY`, `SIGHTENGINE_USER` und `SIGHTENGINE_SECRET` dürfen fehlen. API-Engines überspringen sauber, solange keine Credentials gesetzt sind.
- Windows-Pfade: Pfade mit Leerzeichen in Anführungszeichen setzen, z. B. `python moderate_image.py "C:\Users\me\Pictures\Test Image.PNG" --no-apis`. Backslashes und Großbuchstaben in Erweiterungen werden unterstützt.
