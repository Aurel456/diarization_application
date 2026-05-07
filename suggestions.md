# Améliorations restantes

Ce fichier liste les suggestions d'amélioration **non implémentées** dans cette
itération. Les items 2, 3, 4, 5, 6, 7, 8, 9, 13, 17 et 18 de la proposition
initiale sont déjà appliqués dans le code — voir le commit correspondant.

Priorisation indicative : 🔴 critique · 🟠 forte · 🟡 moyenne.

---

## 🔴 #1 — Tests automatisés

**Constat** : aucun test unitaire ou d'intégration dans le repo. Le bug HDBSCAN
sur audio court (commit précédent) aurait été attrapé par un test e2e sur un
petit WAV de 5s.

**Plan minimum** :
1. `tests/unit/test_clusterer.py` — tests du fallback HDBSCAN (n=1, n=2, n=5), du mapping majority cluster, du plot skip t-SNE.
2. `tests/unit/test_cleaner.py` — LLM mocké via `monkeypatch` sur `LLMClient.chat_text`, vérifie que les exceptions LLM retournent le texte original.
3. `tests/unit/test_config_builder.py` — vérifie que les overrides écrasent bien les settings, que `stateless=True` force `reuse_cache=False`.
4. `tests/unit/test_llm_client.py` — mock OpenAI, vérifie retries, parse JSON avec fences, `wrap_untrusted` échappe les tags.
5. `tests/integration/test_pipeline_short_audio.py` — charge un WAV de 10s dans `fixtures/`, lance le pipeline avec un serveur Whisper/LLM mocké (httpx.MockTransport), vérifie que `run_pipeline` produit un DOCX.

**Outils** : `pytest`, `pytest-mock`, `pytest-asyncio`. Ajouter à
`requirements.txt`. Configurer un workflow GitHub Actions pour les lancer à
chaque PR.

---

## 🟠 #11 — Authentification de l'API FastAPI

**Constat** : `api.py` est totalement ouvert. N'importe qui avec l'URL peut
uploader, déclencher un run (coût GPU/LLM), et télécharger les transcriptions
des autres jobs.

**Plan** : header `X-API-Key` vérifié via dépendance FastAPI.

```python
# api.py
from fastapi import Depends, Header
from settings import settings

async def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("API_PROTECTION_KEY")  # nouvelle var d'env
    if not expected:
        return  # auth désactivée si pas de clé configurée
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.post("/api/v1/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe_audio(...): ...
```

Pour du multi-tenant plus sérieux : OAuth2 Bearer via `fastapi.security`, ou
clé signée JWT. Mais pour un déploiement interne, la clé statique suffit.

Ajouter aussi un rate-limit (`slowapi`) : 1 req / min / IP pour `/transcribe`.

---

## 🟠 #12 — Limite de taille des uploads

**Constat** : `.streamlit/config.toml` autorise 5 GB. Côté FastAPI aucune
limite côté serveur, donc DoS trivial avec un upload géant.

**Plan** :
- FastAPI : utiliser `starlette.middleware` ou lire le header `Content-Length` en amont et rejeter > N MB. Exemple :
```python
MAX_SIZE = int(os.getenv("MAX_UPLOAD_BYTES", 500 * 1024 * 1024))  # 500 MB
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.url.path.startswith("/api/v1/transcribe"):
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_SIZE:
            return JSONResponse(status_code=413, content={"detail": "File too large"})
    return await call_next(request)
```
- Streamlit : baisser `maxUploadSize = 500` dans `config.toml`.

---

## 🟠 #14 — Transcription Whisper en parallèle

**Constat** : `transcribe_all_segments` traite les segments séquentiellement.
Avec un GPU vLLM capable de servir N requêtes simultanées, on sous-utilise la
ressource. Pour une réunion d'1h30 ≈ 400 segments × 0.5s de latence Whisper ≈
3 min de transcription inutilement sérialisés.

**Plan** :
- `concurrent.futures.ThreadPoolExecutor(max_workers=N)` autour de la boucle
  de `transcribe_all_segments`. Thread-safe car chaque segment a son propre
  fichier temp + client OpenAI est thread-safe.
- Paramètre `WHISPER_CONCURRENCY` dans `.env` (default 4).
- Respecter `cancel_event` en annulant les futures non démarrés.

Squelette :
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=concurrency) as pool:
    futures = {pool.submit(transcribe_segment, row, ...): i for i, row in rows}
    for fut in as_completed(futures):
        if cancel_event and cancel_event.is_set():
            for f in futures: f.cancel()
            break
        results[futures[fut]] = fut.result()
```

Important : garder l'ordre de sortie aligné sur l'ordre d'entrée (utiliser
l'index, pas l'ordre de complétion).

---

## 🟡 #15 — Mémoire : stocker les embeddings hors-pickle

**Constat** : `saved_state/step4_clustering_results.pkl` contient la DataFrame
entière, colonne `embedding` incluse (vecteurs 256-D). Pour une réunion avec
10000 segments × 3 representatives × 256 floats32 ≈ 30 MB juste d'embeddings,
re-sérialisés à chaque étape downstream.

**Plan** :
- Stocker les embeddings dans `saved_state/embeddings.npy` (format numpy
  mmap-friendly).
- Dans la DataFrame, ne garder qu'un index ligne → offset dans le `.npy`.
- Pickle plus petit, mmap possible au clustering.

Optionnel : utiliser `zarr` si on veut du versioning ou plusieurs runs.

---

## 🟡 #16 — Jobs persistants (SQLite / Redis)

**Constat** : `api.jobs` et `api.cancel_events` sont en mémoire. Au restart du
conteneur → tous les jobs disparaissent. Le user voulait se mettre en stateless
pour Docker, donc ce n'est pas un problème pour lui **si on accepte que le
client redemande**.

**Plan (si nécessaire plus tard)** :
- SQLite : simple, un fichier persistant monté en volume.
- Redis : si on scale à plusieurs workers, indispensable (file d'attente partagée).
- Schéma :
  ```sql
  CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    audio_path TEXT,
    result_json TEXT,
    created_at TIMESTAMP,
    completed_at TIMESTAMP
  );
  ```
- `cancel_events` resterait en mémoire (local au worker qui exécute).

Pour le mode **stateless actuel** (contraint Docker), ne pas implémenter —
noter seulement que le client doit accepter que `GET /status/{id}` retourne
404 après restart.

---

## Récapitulatif des items implémentés dans ce commit

| # | Description | Fichiers principaux |
|---|-------------|---------------------|
| 2 | Pydantic settings | `settings.py` |
| 3 | Error types spécialisés | `src/errors.py`, `main_pipeline.py` |
| 4 | Cache config-hash + stateless | `src/config_builder.py`, `settings.py`, `main_pipeline.py` |
| 5 | Re-export DOCX avec labels manuels | `src/exporter.py`, `streamlit_app.py` |
| 6 | Streaming transcription (UX + métriques live) | `streamlit_app.py` |
| 7 | Merge manuel de clusters | `src/exporter.py` (`apply_speaker_mapping`), `streamlit_app.py` |
| 8 | Progress per-segment transcription | `src/transcriber.py`, `main_pipeline.py` |
| 9 | Cancellation via `threading.Event` | `main_pipeline.py`, `api.py` (`POST /cancel/{id}`) |
| 13 | Batch embedding computation | `src/clusterer.py` (`compute_embeddings_batch`) |
| 17 | Client LLM unifié | `src/llm_client.py`, refactor `cleaner`/`speaker_identifier`/`summarizer`/`meeting_minutes` |
| 18 | Config builder partagé | `src/config_builder.py`, wired dans `run.py`/`api.py`/`streamlit_app.py` |

---

## Déploiement Docker stateless

Avec `APP_STATELESS=true` dans l'environnement :
- `reuse_cache = False` (pas de pickle cache inter-run)
- `clear_saved_state = True` (nettoyage au démarrage)
- `auto_delete_outputs = True` (nettoyage après run)
- `experiments_root` pointe sur `tempfile.gettempdir()/diarization_experiments`

Dockerfile recommandé (à ajuster selon ton CI) :
```dockerfile
ENV APP_STATELESS=true
ENV APP_ENV=production
ENV APP_AUTO_DELETE_OUTPUTS=true
```

Et monter uniquement `/app/model_storage` en volume (les modèles Pyannote
partagés). Pas besoin de monter `experiments/` ni `saved_state/`.

---

## Limite connue : cancellation Streamlit

`cancel_event` fonctionne quand le pipeline tourne dans un thread séparé (API
FastAPI + `BackgroundTasks`). En Streamlit le pipeline tourne dans le main
thread, donc un bouton "Annuler" ne peut pas être rendu pendant l'exécution.

Solution : envelopper `run_pipeline` dans un `threading.Thread` côté Streamlit
et poller l'état via `st.rerun()` sur un intervalle. Non implémenté ici car
cela change significativement la UX (il faut un état "running" persistant et
une queue pour les logs). À faire en même temps que #1 (tests) pour valider.
