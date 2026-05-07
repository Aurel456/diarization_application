# Diarization Application

Pipeline audio complet : diarization, transcription, nettoyage LLM, identification des locuteurs (automatique **et** manuelle), fusion de clusters, et génération de comptes rendus professionnels.

La **diarization** tourne **en local sur GPU** (Pyannote dans `model_storage/`). La **transcription** et les appels **LLM** passent par un **serveur GPU distant** exposant des endpoints compatibles OpenAI (vLLM ou équivalent).

---

## Architecture

```
Audio (mp3/mp4/wav/…)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         PIPELINE                                │
│                                                                 │
│  1. Audio Splitting       — pydub, chunks de N min              │
│  2. Diarization           — Pyannote (GPU local)                │
│  3. Speaker Clustering    — Pyannote Inference + HDBSCAN        │
│  4. Transcription         — Whisper via API (serveur GPU)       │
│  5. Nettoyage LLM         — via API (optionnel)                 │
│  6. Speaker ID par LLM    — transcription complète → JSON       │
│  7. Export                — DOCX / TXT / SRT                    │
│  8. Compte rendu          — 6 formats (standard → CODIR → RH)   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
experiments/<run_id>/
  ├── output_DOC/
  │   ├── Transcription_Final.docx
  │   ├── Transcription_Final.txt
  │   ├── Transcription_Final.srt
  │   ├── speaker_identification.json
  │   ├── Summary.txt
  │   └── meeting_minutes_<run_id>_minutes.{json,md}
  ├── saved_state/        ← cache intermédiaire (pickle, invalidé auto sur changement de config)
  ├── diarization_results/
  ├── plot/               ← visualisation t-SNE des clusters
  └── logs.txt
```

---

## Modes d'utilisation

| Mode | Commande | Description |
|------|----------|-------------|
| **CLI** | `python run.py` | Lancement direct depuis le terminal |
| **Streamlit** | `streamlit run streamlit_app.py` | Interface web interactive |
| **API REST** | `uvicorn api:app` | Endpoint FastAPI pour intégration |

---

## Installation

### Prérequis

- Python 3.11 ou 3.12
- GPU CUDA local (diarization Pyannote)
- Serveur GPU distant avec vLLM ou équivalent exposant `/v1` (Whisper + LLM)

### Dépendances

```bash
pip install -r requirements.txt
```

### Modèles locaux

```
model_storage/
├── pyannote-speaker-diarization-community-1/   ← pipeline diarization
├── pyannote-wespeaker-voxceleb-resnet34-LM/    ← embedding speaker
└── speechbrain/spkrec-ecapa-voxceleb/          ← fallback optionnel
```

---

## Configuration (`.env`)

```env
# === ENVIRONNEMENT ===
APP_ENV=development           # development | production
APP_STATELESS=false           # true → mode Docker stateless (pas de cache pickle)

# === FICHIERS D'ENTRÉE ===
ROOT=chemin/vers/les/données
INPUT_AUDIO='["fichier1.mp4", "fichier2.qt"]'
AUDIO_PROCESSING_MODE=sequential   # sequential | concurrent

# === DIARIZATION ===
APP_SEGMENT_DURATION=1200     # durée des chunks (secondes)
APP_MAX_WORKERS=2             # parallélisme diarization
APP_MIN_SPEAKER_DURATION=0.5  # durée min d'un segment
APP_VAD_FILTER=true
HF_TOKEN=                     # optionnel

# === HINTS NOMBRE DE LOCUTEURS (optionnel) ===
APP_NUM_SPEAKERS=             # nombre exact connu → AgglomerativeClustering (force N clusters)
APP_MIN_SPEAKERS=             # borne minimum → re-clustering si HDBSCAN trouve moins
APP_MAX_SPEAKERS=             # borne maximum → re-clustering si HDBSCAN trouve plus

# === SERVEUR WHISPER ===
SERVER_URL=http://gpu-server:8000/v1
API_KEY=sk-...
WHISPER_MODEL=whisper

# === LLM (nettoyage, résumé, speaker ID, CR) ===
LLM_MODEL=Gpt-oss-120b
LLM_BASE_URL=http://gpu-server:8000/v1

# === FEATURES OPTIONNELLES ===
APP_ENABLE_LLM_CLEANING=true        # nettoyage du texte (orthographe, ponctuation, hésitations)
APP_ENABLE_SUMMARY=true             # résumé global du contenu (indépendant du cleaning)
APP_ENABLE_SPEAKER_IDENTIFICATION=false
APP_ENABLE_MEETING_MINUTES=false
SPEAKER_IDENTIFICATION_MODEL=
MEETING_MINUTES_MODEL=

# === CACHE ET NETTOYAGE ===
APP_REUSE_CACHE=true
APP_FORCE_SPLIT=false
APP_CLEAR_SAVED_STATE=false
APP_AUTO_DELETE_OUTPUTS=false
APP_CLEANUP_GRACE_SECONDS=10800

# === LOGS ===
APP_LOG_LEVEL=INFO
APP_LOG_TO_CONSOLE=true
APP_HIDE_LOG_PANEL=false
```

Toute la configuration passe par **Pydantic `BaseSettings`** : validation typée des valeurs, defaults env-aware, messages d'erreur explicites.

---

## Utilisation — CLI (`run.py`)

```bash
# Config 100 % depuis .env
python run.py

# Fichiers en direct
python run.py --audio reunion.mp4

# Plusieurs fichiers consécutifs
python run.py --audio partie1.mp4 partie2.mp4 --mode sequential

# Activer ID locuteurs + CR
python run.py --audio reunion.mp4 --speaker-id --meeting-minutes

# Format de CR spécifique
python run.py --audio reunion.mp4 --meeting-minutes --meeting-minutes-format executif

# Transcription simple (sans diarization)
python run.py --audio reunion.mp4 --simple

# Lancer l'API FastAPI
python run.py --serve --port 8000

# Aide complète
python run.py --help
```

### Options CLI

| Option | Description |
|--------|-------------|
| `--audio FILE [FILE...]` | Fichiers audio/vidéo |
| `--root DIR` | Répertoire racine |
| `--run-id ID` | Identifiant du run |
| `--mode sequential\|concurrent` | Multi-fichiers |
| `--simple` | Transcription simple (pas de diarization) |
| `--serve` / `--host` / `--port` | Lance l'API FastAPI |
| `--segment-duration SEC` | Durée des chunks |
| `--max-workers N` | Parallélisme diarization |
| `--no-vad` | Désactive le VAD |
| `--server-url URL` / `--whisper-model NAME` / `--api-key KEY` | Config Whisper |
| `--llm-url URL` / `--llm-model NAME` | Config LLM |
| `--no-cleaning` | Désactive le nettoyage LLM |
| `--speaker-id` / `--speaker-id-model NAME` | ID locuteurs par LLM |
| `--meeting-minutes` / `--meeting-minutes-model NAME` | CR par LLM |
| `--meeting-minutes-format KEY` | `standard`, `executif`, `technique`, `projet`, `rh_social`, `formation` |
| `--no-cache` | Ignore le cache |
| `--experiments-dir DIR` | Dossier de sortie |
| `--log-level LEVEL` | DEBUG / INFO / WARNING / ERROR |

---

## Utilisation — Streamlit

```bash
streamlit run streamlit_app.py
```

Accès sur `http://localhost:8501`.

### 5 modes de travail

| Mode | Description |
|------|-------------|
| **Transcription rapide** | Whisper seul, sans diarization |
| **Sous-titres (SRT)** | Génération SRT horodaté |
| **Détection des locuteurs + compte rendu** | Pipeline complet avec toutes les features |
| **🎙️ Enregistrement Live** | Capture micro + transcription instantanée + lancement diarization direct (sans re-upload) |
| **🎙️ Streaming temps réel** | WebRTC continu : audio bufferisé en chunks de 3-15 s, chaque chunk envoyé à Whisper, transcription qui s'auto-rafraîchit pendant la capture |

### Contrôles & boutons (Full Pipeline)

| Section | Contrôle | Rôle |
|---------|----------|------|
| **Header** | Sélecteur `Tâche` (mode) + `Stratégie` (batch/séquentiel/simultané) | Choix du scénario |
| **Sidebar** | `Upload audio files` | Un ou plusieurs fichiers |
| **Sidebar** | `Run ID prefix` | Nom du dossier de sortie |
| **Sidebar** | Paramètres chunk / VAD / workers | Réglages pipeline |
| **Sidebar** | **🎯 Nombre de locuteurs** : `Inconnu` / `Connu exactement` / `Plage min-max` | Hint pour le clustering (voir §2 Clustering) |
| **Sidebar** | **🤖 Traitement LLM** : `LLM Cleaning` + `Résumé` (cases séparées) | Active indépendamment chaque feature LLM |
| **Sidebar** | `Identifier les locuteurs via LLM` | Active l'ID auto par LLM |
| **Sidebar** | `Générer un compte rendu` + format + instructions | Active le CR pipeline |
| **Sidebar** | `Vérifier la connectivité` | Health check Whisper + LLM |
| **Sidebar** | `⚠️ Reset / Unblock State` | Secours si UI bloquée |
| **Main** | `🚀 Lancer le traitement` | Démarre le pipeline |
| **Main** | `Effacer l'erreur` | Masque une erreur persistante |
| **Main** | **📁 Fichiers générés** (cases à cocher) | Sélection des sorties à télécharger (DOCX/TXT/SRT/Résumé/Speaker JSON/CR) |

### Identification des locuteurs par l'utilisateur (après diarization)

La section **🎧 Identification manuelle des locuteurs** permet de :

| Élément | Comportement |
|---------|--------------|
| **2 extraits audio par locuteur** | Écouter les 2 segments les plus longs pour reconnaître la voix |
| **Aperçu de 3 phrases** | Lire des extraits de transcription pour recouper |
| **Champs Prénom / Nom / Fonction** | Saisir manuellement l'identité de chaque locuteur |
| **Pré-remplissage LLM** | Si `APP_ENABLE_SPEAKER_IDENTIFICATION=true`, les champs affichent les suggestions du LLM par défaut |
| **⬇️ Exporter les labels (JSON)** | Sauvegarde du mapping saisi |
| **♻️ Réinitialiser depuis LLM** | Remet les formulaires aux suggestions LLM (ou vide) |

### Fusion de clusters (corrections HDBSCAN)

HDBSCAN se trompe parfois (sur-segmente une même personne en plusieurs clusters). La section **🔗 Fusionner des clusters** permet de corriger :

1. Multiselect des clusters à fusionner
2. Choix du cluster cible
3. Bouton `Appliquer la fusion` → les segments sont renommés
4. `🗑️ Réinitialiser les fusions` annule toutes les fusions actives

### Regénération DOCX

Une fois les labels saisis et/ou les fusions appliquées :

- **📄 Regénérer le DOCX avec labels + fusions** : reconstruit un DOCX avec les vrais noms et le mapping de fusion, **sans relancer** diarization/transcription/cleaning.

### Génération de compte rendu à la volée

La section **📋 Générer un compte rendu** permet de (re)générer un CR :

- **Source des locuteurs** (radio) : `Sans identification` / `Identification LLM automatique` / `Labels manuels (saisis ci-dessus)`
- **Format** : 6 templates (standard, exécutif CODIR, technique IT, projet agile, RH/CSE, formation)
- **Titre** / **Date** (optionnels) injectés dans le prompt
- **Instructions spécifiques** texte libre
- Boutons `⬇️ Télécharger Markdown` et `⬇️ Télécharger JSON`
- Aperçu Markdown intégré

Les fusions de clusters actives sont appliquées automatiquement au transcript envoyé au LLM.

### Enregistrement Live (🎙️)

1. `Enregistrez votre intervention` (st.audio_input)
2. Transcription automatique dès la fin de la capture (Whisper)
3. Accumulation dans un historique déroulant avec métriques temps-réel (interventions / mots / caractères)
4. **`🔬 Lancer la diarization sur cet audio`** fusionne tous les clips et lance **directement** le pipeline complet — **pas de re-upload manuel requis**
5. `🗑️ Tout effacer` remet l'historique à zéro
6. `⬇️ Télécharger la transcription (.txt)` exporte le texte accumulé

Les enregistrements sont dédupliqués (`hash(audio_bytes)`) — re-soumettre le même clip n'est pas transcrit deux fois.

### Streaming temps réel (🎙️)

Capture continue via **WebRTC** (streamlit-webrtc) : pas besoin d'arrêter et relancer le micro entre chaque phrase.

| Élément | Comportement |
|---------|--------------|
| **Slider `Durée des chunks`** | 3 à 15 s — durée d'audio accumulée avant chaque envoi à Whisper |
| **Bouton `START`** | Démarre la capture WebRTC (le navigateur demande l'accès micro) |
| **Bouton `STOP`** | Stoppe la capture |
| **🗑️ `Effacer`** | Réinitialise l'historique de transcription |
| **Auto-refresh** | La page se rafraîchit toutes les ~0.8 s tant que le stream est actif |
| **⬇️ Télécharger (.txt)** | Exporte la transcription accumulée |

Pipeline interne :
1. `WhisperAudioProcessor` (sous-classe de `AudioProcessorBase`) reçoit chaque `av.AudioFrame`
2. Conversion `fltp` → `int16`, mixage stéréo → mono
3. Buffering jusqu'à atteindre la durée de chunk
4. Background thread : ré-échantillonnage à 16 kHz → WAV → Whisper API
5. Résultat poussé dans une `queue.Queue` consommée par le main thread Streamlit

Le streaming n'a pas de diarization (juste transcription continue). Pour faire la diarization sur un live, utiliser le mode **Enregistrement Live** qui propose le bouton "Lancer la diarization sur cet audio".

### Health Check

La barre latérale affiche l'état des serveurs Whisper et LLM :

- **✅ OK** — serveur accessible, modèle trouvé, latence en ms
- **⚠️ Connecté** — serveur OK, modèle introuvable (liste dispo affichée)
- **❌ Inaccessible** — erreur connexion

Si les deux endpoints sont identiques, un seul appel `/models` est fait (dedup).

---

## Utilisation — API REST (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# ou
python run.py --serve --port 8000
```

Docs interactives : `http://localhost:8000/docs`.

### Endpoints

| Méthode | URL | Description |
|---------|-----|-------------|
| `GET` | `/` | Info API |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/transcribe` | Upload audio → démarre un job |
| `GET` | `/api/v1/status/{job_id}` | Statut (`queued`/`processing`/`completed`/`failed`/`cancelled`) |
| `POST` | `/api/v1/cancel/{job_id}` | Annulation gracieuse du job en cours |
| `GET` | `/api/v1/download/{job_id}/{type}` | Télécharger un résultat |
| `DELETE` | `/api/v1/job/{job_id}` | Supprimer un job et ses fichiers |

### Types téléchargeables

`docx` · `txt` · `srt` · `summary` · `speaker_json` · `json` (CR) · `markdown` (CR)

### Exemple

```bash
# Lancer
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@reunion.mp4" \
  -F "enable_speaker_identification=true" \
  -F "enable_meeting_minutes=true"
# → {"job_id": "abc-123", "status": "queued", ...}

# Suivre
curl http://localhost:8000/api/v1/status/abc-123

# Annuler
curl -X POST http://localhost:8000/api/v1/cancel/abc-123

# Télécharger
curl -o result.docx http://localhost:8000/api/v1/download/abc-123/docx
```

---

## Déploiement Docker — mode stateless

Pour un déploiement sans persistance inter-containers, activer le mode stateless :

```dockerfile
ENV APP_STATELESS=true
ENV APP_ENV=production
```

En mode stateless :
- `reuse_cache=False` (pas de pickle inter-run)
- `clear_saved_state=True` (nettoyage au démarrage)
- `auto_delete_outputs=True` (nettoyage après run)
- `auto_delete_uploads=True`
- `experiments_root` pointé sur `tempfile.gettempdir()/diarization_experiments`

Seul le volume `model_storage/` doit être monté (modèles Pyannote partagés).

```bash
docker build -t diarization-app .

# Streamlit
docker run -p 8501:8501 --gpus all \
  -e APP_STATELESS=true \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/model_storage:/app/model_storage \
  diarization-app

# API REST
docker run -p 8000:8000 --gpus all \
  -e APP_STATELESS=true \
  -v $(pwd)/.env:/app/.env \
  -v $(pwd)/model_storage:/app/model_storage \
  diarization-app \
  python run.py --serve --port 8000
```

---

## Fonctionnalités détaillées

### 1. Diarization (local GPU)

Pyannote Audio segmente l'audio et identifie les tours de parole. Chaque chunk de N minutes est traité en parallèle, puis les résultats sont fusionnés.

### 2. Clustering des locuteurs

Embeddings (`wespeaker-voxceleb-resnet34-LM`) groupés par **HDBSCAN** pour obtenir des identités globales cohérentes (`Speaker_00`, `Speaker_01`, …) sur tout l'audio.

**Calcul batché** : les embeddings sont calculés en chargeant chaque fichier chunk **une seule fois** puis en le croppant pour tous ses segments — plusieurs fois plus rapide sur les longs audios.

**Fallback robuste** : audio court avec 1–2 segments → assignation à un cluster unique au lieu de crasher.

**Hints du nombre de locuteurs** (intelligemment liés à HDBSCAN) :

| Hint | Comportement | Méthode |
|------|--------------|---------|
| `APP_NUM_SPEAKERS=N` | Force exactement N clusters | **AgglomerativeClustering** (cosine, average linkage) — bypasse HDBSCAN |
| `APP_MIN_SPEAKERS` / `APP_MAX_SPEAKERS` | Borne le résultat HDBSCAN | HDBSCAN d'abord, puis re-clustering AgglomerativeClustering avec `n_clusters` clampé si hors bornes |
| Aucun hint | Auto | HDBSCAN pur (comportement historique) |

Le hint n'est **pas** envoyé aux appels Pyannote per-chunk (chaque chunk ne voit qu'un fragment de l'audio — forcer N par chunk sur-segmenterait). La contrainte s'applique uniquement à l'étape de clustering global.

`clustering_details["clustering_method"]` enregistre quelle méthode a été utilisée : `agglomerative_exact` / `agglomerative_clamped` / `hdbscan` / `single_cluster`.

Visualisation t-SNE sauvegardée dans `plot/`.

### 3. Transcription (serveur GPU)

Envoi OpenAI-compat vers Whisper. Retry automatique avec backoff. **Progression par segment** rapportée à l'UI (compteur `done/total`).

### 4. Nettoyage LLM (optionnel)

Le LLM corrige orthographe, ponctuation, casse, supprime les bégaiements et hésitations, sans modifier le fond. Protection anti-prompt-injection : le texte brut est encadré de balises `<untrusted_input>` et le système rappelle au LLM de ne pas suivre d'instructions injectées.

Contrôlé par `APP_ENABLE_LLM_CLEANING` / `--no-cleaning`. **Indépendant du résumé** (`APP_ENABLE_SUMMARY`) — chaque feature peut être activée/désactivée séparément. Lorsque le cleaning est désactivé, l'étape est entièrement skippée (status `skipped` dans la barre de progression) et la transcription brute est utilisée pour les étapes suivantes.

### 5. Identification des locuteurs par LLM (optionnel)

Après le nettoyage, le LLM reçoit la **transcription chronologique complète** avec timestamps et labels, puis déduit nom/prénom/fonction de chaque intervenant. Transcript passé en bloc `<transcript>…</transcript>` pour limiter l'injection de prompt.

Sortie JSON :
```json
{
  "SPEAKER_00": {
    "speaker_id": "SPEAKER_00",
    "nom": "MARTIN",
    "prenom": "Sophie",
    "fonction": "Directrice de projet",
    "confidence": 0.92
  }
}
```

Contrôlé par `APP_ENABLE_SPEAKER_IDENTIFICATION` / `--speaker-id`.

### 6. Identification manuelle + fusion (dans Streamlit)

Voir sections [Identification des locuteurs par l'utilisateur](#identification-des-locuteurs-par-lutilisateur-après-diarization) et [Fusion de clusters](#fusion-de-clusters-corrections-hdbscan).

### 7. Export (DOCX / TXT / SRT)

DOCX avec locuteurs identifiés, timestamps, mise en forme. Les noms réels remplacent les labels si l'ID est active.

`re_export_docx_with_labels()` permet de régénérer le DOCX **après** le pipeline avec de nouveaux labels ou des fusions appliquées.

### 8. Compte rendu de réunion

6 formats disponibles :

| Clé | Label | Usage |
|-----|-------|-------|
| `standard` | Standard | Compte rendu classique |
| `executif` | Exécutif / CODIR | Synthèse décideurs |
| `technique` | Technique / IT | Tickets, solutions, dépendances |
| `projet` | Projet / Agile | Sprint, blockers, jalons |
| `rh_social` | RH / Dialogue social | CSE, NAO |
| `formation` | Formation / Séminaire | Workshop, retours |

Sauvegardé en **JSON** et **Markdown**. Contrôlé par `APP_ENABLE_MEETING_MINUTES` / `--meeting-minutes`.

### 9. Client LLM unifié

`src/llm_client.py` centralise toutes les interactions LLM :
- Retries exponentiels automatiques
- Parsing JSON tolérant (fences markdown, extraction regex)
- Accounting des tokens (`client.stats`)
- Helper `wrap_untrusted()` contre les prompt injections
- Gestion d'erreurs typées via `LLMError`

### 10. Cancellation gracieuse

`cancel_event: threading.Event` attaché au `PipelineConfig` :
- Côté API : `POST /api/v1/cancel/{job_id}` set l'event
- Pipeline : checkpoints entre chaque étape (`_check_cancel`) lèvent `CancelledError`
- Transcription : vérifié entre chaque segment
- Statut du job passe à `cancelled` (distinct de `failed`)

### 11. Health check

`src/health_check.py` vérifie disponibilité + latence Whisper/LLM. Si les deux endpoints sont identiques, dedup de l'appel `/models`.

### 12. Cache intelligent + mode stateless

- `saved_state/.config_hash` stocke le hash des paramètres qui affectent le pipeline (`segment_duration`, `llm_model`, etc.). Si le hash change, tous les pickles sont invalidés automatiquement — fini les résultats stale après un changement de config.
- Mode `APP_STATELESS=true` : désactive entièrement le cache, force le cleanup, experiments dans `tmp/` — adapté au déploiement Docker.

### 13. Types d'erreurs spécialisés

`src/errors.py` expose une hiérarchie :

```
PipelineError
├── ConfigurationError
├── AudioInputError
├── AudioSplittingError
├── DiarizationError
├── ClusteringError
├── TranscriptionError   (retryable=True)
├── LLMError             (retryable=True)
├── ExportError
└── CancelledError
```

Chaque erreur connaît son `step` et un flag `retryable` — les clients API peuvent choisir de retry seulement les erreurs marquées.

---

## Structure du projet

```
diarization_application/
├── run.py                    ← CLI entry point (argparse)
├── main_pipeline.py          ← Orchestrateur pipeline
├── api.py                    ← Serveur FastAPI
├── streamlit_app.py          ← Interface Streamlit
├── settings.py               ← Config Pydantic BaseSettings
├── .env                      ← Variables d'environnement
│
├── src/
│   ├── config_builder.py     ← Constructeur PipelineConfig partagé + cache hash
│   ├── errors.py             ← Types d'erreurs spécialisés
│   ├── llm_client.py         ← Client LLM unifié (retries, JSON, stats)
│   ├── audio_splitter.py     ← Découpage audio
│   ├── diarizer.py           ← Diarization Pyannote
│   ├── clusterer.py          ← Clustering (HDBSCAN) + embeddings batchés
│   ├── transcriber.py        ← Whisper API + progression per-segment
│   ├── cleaner.py            ← Nettoyage LLM
│   ├── speaker_identifier.py ← ID locuteurs par LLM
│   ├── speaker_audio_sampler.py ← Extraction clips audio
│   ├── meeting_minutes.py    ← Génération CR (6 formats)
│   ├── health_check.py       ← Vérif serveurs Whisper/LLM
│   ├── exporter.py           ← DOCX / TXT / SRT + re-export manuel + merge clusters
│   ├── summarizer.py         ← Résumé LLM
│   ├── simple_transcriber.py ← Transcription simple
│   └── utils.py              ← Utilitaires
│
├── model_storage/            ← Modèles Pyannote
├── tests/                    ← Suite de tests unitaires (pytest)
│   ├── conftest.py           ← Fixtures partagées
│   ├── test_errors.py
│   ├── test_exporter.py
│   ├── test_meeting_minutes.py
│   ├── test_speaker_identifier.py
│   ├── test_llm_client.py
│   ├── test_config_builder.py
│   ├── test_utils.py
│   ├── test_simple_transcriber.py
│   ├── test_health_check.py
│   └── test_summarizer.py
├── .streamlit/config.toml
├── Dockerfile
├── compose.yml
├── requirements.txt
├── suggestions.md            ← Améliorations restantes
└── README.md
```

---

## Variables d'environnement — référence complète

| Variable | Défaut | Description |
|----------|--------|-------------|
| `APP_ENV` | `development` | `development` / `production` |
| `APP_STATELESS` | `false` | Mode stateless Docker |
| `ROOT` | `.` | Racine des données |
| `EXPERIMENTS_ROOT` | `./experiments` (ou tmp en stateless) | Dossier de sortie |
| `INPUT_AUDIO` | — | Liste JSON des fichiers |
| `AUDIO_PROCESSING_MODE` | `concurrent` | `sequential` / `concurrent` |
| `APP_SEGMENT_DURATION` | `1200` | Durée chunks (s) |
| `APP_MAX_WORKERS` | `2` | Workers diarization |
| `APP_MIN_SPEAKER_DURATION` | `0.5` | Durée min segment (s) |
| `APP_VAD_FILTER` | `true` | Filtre VAD |
| `HF_TOKEN` | — | Token HuggingFace |
| `SERVER_URL` | — | URL Whisper (`/v1`) |
| `WHISPER_MODEL` | `whisper` | Modèle Whisper |
| `API_KEY` | — | Clé API serveur |
| `LLM_MODEL` | — | Modèle LLM |
| `LLM_BASE_URL` | — | URL LLM (`/v1`) |
| `APP_ENABLE_LLM_CLEANING` | `true` (dev) | Nettoyage LLM (texte propre) |
| `APP_ENABLE_SUMMARY` | `true` (dev) | Résumé LLM (indépendant du cleaning) |
| `APP_NUM_SPEAKERS` | — | Nombre exact de locuteurs (force AgglomerativeClustering) |
| `APP_MIN_SPEAKERS` | — | Borne min — re-clustering si HDBSCAN trouve moins |
| `APP_MAX_SPEAKERS` | — | Borne max — re-clustering si HDBSCAN trouve plus |
| `APP_ENABLE_SPEAKER_IDENTIFICATION` | `false` | ID locuteurs |
| `SPEAKER_IDENTIFICATION_MODEL` | — | Modèle dédié ID |
| `APP_ENABLE_MEETING_MINUTES` | `false` | CR |
| `MEETING_MINUTES_MODEL` | — | Modèle dédié CR |
| `APP_REUSE_CACHE` | `true` (dev) / `false` (stateless) | Réutilise pickle cache |
| `APP_FORCE_SPLIT` | `false` | Force re-découpage |
| `APP_CLEAR_SAVED_STATE` | `false` (dev) / `true` (stateless) | Efface cache au start |
| `APP_AUTO_DELETE_OUTPUTS` | `false` (dev) | Suppression auto post-run |
| `APP_CLEANUP_GRACE_SECONDS` | `10800` | Délai avant suppression |
| `APP_LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |
| `APP_LOG_TO_CONSOLE` | `true` (dev) | Logs console |
| `APP_HIDE_LOG_PANEL` | `false` (dev) | Masque panneau Streamlit |

---

## Tests

Suite de **162 tests unitaires** (pytest), exécutables sans GPU ni réseau :

```bash
pip install pytest pytest-mock
python -m pytest tests/ -v
```

| Fichier | Couverture |
|---------|------------|
| `test_errors.py` | Hiérarchie d'erreurs — constructeurs, `cause`, `retryable`, `step`, `__str__` |
| `test_exporter.py` | `split_sentences_with_linebreaks`, `clean_before_export`, `concatenate_texts`, `apply_speaker_mapping` |
| `test_meeting_minutes.py` | Dataclasses, sérialisation JSON, `minutes_to_markdown`, 6 formats, `_build_transcript_block` |
| `test_speaker_identifier.py` | `SpeakerInfo`, `_build_chronological_transcript`, `_build_per_speaker_summary`, prompt builders, JSON round-trip |
| `test_llm_client.py` | `UsageStats`, `wrap_untrusted`, `make_llm_client` |
| `test_config_builder.py` | `compute_config_hash`, `ensure_cache_consistency`, invalidation de cache |
| `test_utils.py` | `load_or_run` (cache hit/miss/corrupt), `adjust_timestamps_for_sequential_audio` |
| `test_simple_transcriber.py` | `format_timestamp_srt`, `words_to_srt_blocks`, `save_files` |
| `test_health_check.py` | `ServiceStatus.label`, déduplication `check_all_services` |
| `test_summarizer.py` | `_token_count`, `_chunk_text_by_tokens` |

Les fixtures partagées (`conftest.py`) fournissent des DataFrames, `SpeakerInfo` et `MeetingMinutes` de test prêts à l'emploi.

## Roadmap

### Implémenté

- ✅ Pydantic `BaseSettings` + cache hash-keyed
- ✅ Hiérarchie d'erreurs typées avec `retryable`
- ✅ Cancellation gracieuse via `threading.Event`
- ✅ Client LLM unifié (retries, JSON parsing, token accounting, prompt-injection guard)
- ✅ Re-export DOCX avec labels manuels + fusion clusters
- ✅ Streaming temps réel WebRTC (capture continue micro → Whisper)
- ✅ Progression par segment (transcription)
- ✅ Mode stateless Docker (no cache, tmp dirs)
- ✅ Sélection des fichiers de sortie (cases à cocher dans l'interface)
- ✅ Hints du nombre de locuteurs (exact / min / max) liés intelligemment à HDBSCAN
- ✅ Cleaning et résumé séparés (deux options indépendantes)

### À faire

Voir [suggestions.md](suggestions.md) :

- Auth API (header `X-API-Key`)
- Rate limiting
- Transcription parallèle (asyncio / thread pool)
- Jobs persistants en SQLite/Redis (alternative au mode stateless)
- Annulation par segment plus granulaire
