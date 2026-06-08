# api.py
"""
FastAPI application exposing audio processing pipeline as REST endpoints.
"""

import os
import uuid
import logging
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from main_pipeline import (
    run_pipeline,
    PipelineConfig,
    PipelineResult,
)
from settings import settings
from src.health_check import check_all_services

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Processing API",
    description="REST API for speaker diarization, transcription, and meeting minutes generation",
    version="1.0.0",
)

# In-memory job store (for production, use Redis or DB)
jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
# One cancellation event per job
cancel_events: Dict[str, threading.Event] = {}

_ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".mkv"}
_VALID_MM_FORMATS = {"standard", "executif", "technique", "projet", "rh_social", "formation"}


# --- Pydantic models ---

class JobStatus(BaseModel):
    """Response model for job status."""
    job_id: str
    status: str  # queued | processing | completed | failed | cancelled
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


# --- Background task ---

def process_audio_job(job_id: str, audio_path: str, config_overrides: Dict[str, Any]):
    """Run the audio processing pipeline in a background thread."""
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        logger.info(f"Starting job {job_id} for audio: {audio_path}")

        from src.config_builder import build_pipeline_config as _build

        overrides = {k: v for k, v in config_overrides.items() if v is not None}
        # Drop zero/negative values for numeric params (Swagger UI sends 0 by default)
        for _key in ("segment_duration", "max_workers"):
            if overrides.get(_key, 1) <= 0:
                overrides.pop(_key, None)
        overrides["run_id"] = overrides.get("run_id") or f"api_{job_id}"
        overrides["cancel_event"] = cancel_events.setdefault(job_id, threading.Event())

        base_config = _build(audio_paths=[audio_path], overrides=overrides)
        result = run_pipeline(base_config)

        result_dict = {
            "success": result.success,
            "run_id": result.run_id,
            "experiments_dir": result.experiments_dir,
            "num_speakers_final": result.num_speakers_final,
            "message": result.message,
            "docx_path": result.docx_path,
            "txt_path": result.txt_path,
            "srt_path": result.srt_path,
            "summary_path": result.summary_path,
            "speaker_identification_path": result.speaker_identification_path,
            "speaker_info": result.speaker_info,
            "meeting_minutes_paths": result.meeting_minutes_paths,
        }

        with jobs_lock:
            jobs[job_id].update({
                "status": "completed",
                "result": result_dict,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        from src.errors import CancelledError, PipelineError
        final_status = "cancelled" if isinstance(e, CancelledError) else "failed"
        with jobs_lock:
            jobs[job_id].update({
                "status": final_status,
                "message": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
    finally:
        cancel_events.pop(job_id, None)
        if settings.stateless:
            try:
                up = Path(audio_path)
                if up.exists():
                    up.unlink(missing_ok=True)
                    logger.info(f"Stateless cleanup: removed upload {up}")
            except Exception as cleanup_exc:
                logger.warning(f"Failed to remove upload {audio_path}: {cleanup_exc}")


# --- Shared upload helper ---

async def _upload_and_queue(
    file: UploadFile,
    config_overrides: Dict[str, Any],
    background_tasks: BackgroundTasks,
) -> JobStatus:
    """Validate + save file, register job, queue background task."""
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Type de fichier non supporté. Acceptés : {sorted(_ALLOWED_EXTENSIONS)}",
        )

    job_id = str(uuid.uuid4())
    upload_dir = (
        Path(tempfile.gettempdir()) / "diarization_api_uploads"
        if settings.stateless
        else Path("api_uploads")
    )
    upload_dir.mkdir(parents=True, exist_ok=True)
    audio_path = upload_dir / f"{job_id}{file_ext}"

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Échec de la sauvegarde du fichier")

    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "audio_path": str(audio_path),
            "original_filename": file.filename,
        }

    background_tasks.add_task(process_audio_job, job_id, str(audio_path), config_overrides)
    return JobStatus(job_id=job_id, status="queued", created_at=jobs[job_id]["created_at"])


# --- Endpoints ---

@app.get("/")
def root():
    return {"message": "Audio Processing API", "version": "1.0.0"}


@app.post(
    "/api/v1/transcribe/simple",
    response_model=JobStatus,
    summary="Transcription rapide",
    description="Whisper uniquement, sans diarization. Retourne TXT + SRT.",
    tags=["Transcription"],
)
async def transcribe_simple(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None, description="Modèle Whisper (défaut depuis .env)"),
    enable_summary: Optional[bool] = Form(None, description="Générer un résumé LLM"),
    language: Optional[str] = Form(None, description="Langue audio ISO 639-1 (ex: fr, en)"),
):
    return await _upload_and_queue(file, {
        "run_id": run_id,
        "simple_mode": True,
        "enable_llm_cleaning": False,
        "enable_summary": enable_summary,
        "enable_speaker_identification": False,
        "enable_meeting_minutes": False,
        "whisper_model": whisper_model,
        "language": language,
    }, background_tasks)


@app.post(
    "/api/v1/transcribe/srt",
    response_model=JobStatus,
    summary="Sous-titres (SRT)",
    description="Transcription horodatée avec chunks fins pour la précision des sous-titres. Retourne SRT + TXT.",
    tags=["Transcription"],
)
async def transcribe_srt(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None, description="Modèle Whisper (défaut depuis .env)"),
    language: Optional[str] = Form(None, description="Langue audio ISO 639-1 (ex: fr, en)"),
):
    return await _upload_and_queue(file, {
        "run_id": run_id,
        "simple_mode": True,
        "enable_llm_cleaning": False,
        "enable_summary": False,
        "enable_speaker_identification": False,
        "enable_meeting_minutes": False,
        "whisper_model": whisper_model,
        "language": language,
        "chunk_size": 35,  # chunks fins pour horodatage SRT précis
    }, background_tasks)


@app.post(
    "/api/v1/transcribe/diarize",
    response_model=JobStatus,
    summary="Détection des locuteurs",
    description=(
        "Pipeline complet : diarization + transcription + LLM optionnel. "
        "Retourne DOCX, TXT, SRT, et optionnellement résumé, identification des locuteurs, compte rendu."
    ),
    tags=["Diarization"],
)
async def transcribe_diarize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    segment_duration: Optional[int] = Form(None, ge=1, description="Durée des chunks audio en secondes (défaut : 1200)"),
    max_workers: Optional[int] = Form(None, ge=1, description="Workers parallèles pour la diarization (défaut : 2)"),
    num_speakers: Optional[int] = Form(None, ge=1, description="Nombre exact de locuteurs (optionnel)"),
    min_speakers: Optional[int] = Form(None, ge=1, description="Borne minimum de locuteurs (optionnel)"),
    max_speakers: Optional[int] = Form(None, ge=1, description="Borne maximum de locuteurs (optionnel)"),
    enable_llm_cleaning: Optional[bool] = Form(None, description="Nettoyage LLM (orthographe, ponctuation)"),
    enable_summary: Optional[bool] = Form(None, description="Générer un résumé LLM"),
    enable_speaker_identification: Optional[bool] = Form(None, description="Identifier les locuteurs via LLM"),
    enable_meeting_minutes: Optional[bool] = Form(None, description="Générer un compte rendu"),
    meeting_minutes_format: Optional[str] = Form(None, description="standard | executif | technique | projet | rh_social | formation"),
    whisper_model: Optional[str] = Form(None, description="Modèle Whisper (défaut depuis .env)"),
    language: Optional[str] = Form(None, description="Langue audio ISO 639-1 (ex: fr, en)"),
):
    if meeting_minutes_format and meeting_minutes_format not in _VALID_MM_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"meeting_minutes_format invalide. Valeurs acceptées : {sorted(_VALID_MM_FORMATS)}",
        )
    return await _upload_and_queue(file, {
        "run_id": run_id,
        "simple_mode": False,
        "segment_duration": segment_duration,
        "max_workers": max_workers,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "enable_llm_cleaning": enable_llm_cleaning,
        "enable_summary": enable_summary,
        "enable_speaker_identification": enable_speaker_identification,
        "enable_meeting_minutes": enable_meeting_minutes,
        "meeting_minutes_format": meeting_minutes_format,
        "whisper_model": whisper_model,
        "language": language,
    }, background_tasks)


@app.post(
    "/api/v1/transcribe",
    response_model=JobStatus,
    summary="Endpoint générique (tous paramètres)",
    description="Endpoint complet avec tous les paramètres disponibles. Préférer les endpoints spécialisés ci-dessus.",
    tags=["Générique"],
)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    simple_mode: Optional[bool] = Form(None, description="Transcription simple sans diarization"),
    segment_duration: Optional[int] = Form(None, ge=1, description="Durée des chunks audio en secondes (défaut : 1200)"),
    max_workers: Optional[int] = Form(None, ge=1, description="Workers parallèles pour la diarization (défaut : 2)"),
    num_speakers: Optional[int] = Form(None, ge=1, description="Nombre exact de locuteurs (optionnel)"),
    min_speakers: Optional[int] = Form(None, ge=1, description="Borne minimum de locuteurs (optionnel)"),
    max_speakers: Optional[int] = Form(None, ge=1, description="Borne maximum de locuteurs (optionnel)"),
    enable_llm_cleaning: Optional[bool] = Form(None),
    enable_summary: Optional[bool] = Form(None),
    enable_speaker_identification: Optional[bool] = Form(None),
    enable_meeting_minutes: Optional[bool] = Form(None),
    meeting_minutes_format: Optional[str] = Form(None, description="standard | executif | technique | projet | rh_social | formation"),
    whisper_model: Optional[str] = Form(None, description="Modèle Whisper (défaut depuis .env)"),
    language: Optional[str] = Form(None, description="Langue audio ISO 639-1 (ex: fr, en)"),
):
    if meeting_minutes_format and meeting_minutes_format not in _VALID_MM_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"meeting_minutes_format invalide. Valeurs acceptées : {sorted(_VALID_MM_FORMATS)}",
        )
    return await _upload_and_queue(file, {
        "run_id": run_id,
        "simple_mode": simple_mode,
        "segment_duration": segment_duration,
        "max_workers": max_workers,
        "num_speakers": num_speakers,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "enable_llm_cleaning": enable_llm_cleaning,
        "enable_summary": enable_summary,
        "enable_speaker_identification": enable_speaker_identification,
        "enable_meeting_minutes": enable_meeting_minutes,
        "meeting_minutes_format": meeting_minutes_format,
        "whisper_model": whisper_model,
        "language": language,
    }, background_tasks)


@app.get("/api/v1/status/{job_id}", response_model=JobStatus, tags=["Jobs"])
def get_job_status(job_id: str):
    """Statut d'un job (queued → processing → completed / failed / cancelled)."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("message"),
        result=job.get("result"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


@app.get("/api/v1/download/{job_id}/{file_type}", tags=["Jobs"])
def download_result(job_id: str, file_type: str):
    """
    Télécharger un fichier résultat.

    file_type : docx | txt | srt | summary | speaker_json | standard | executif | technique | projet | rh_social | formation
    """
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    result = job.get("result", {})
    mapping = {
        "docx": result.get("docx_path"),
        "txt": result.get("txt_path"),
        "srt": result.get("srt_path"),
        "summary": result.get("summary_path"),
        "speaker_json": result.get("speaker_identification_path"),
    }
    meeting_paths = result.get("meeting_minutes_paths", {})
    file_path = meeting_paths.get(file_type) or mapping.get(file_type)

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"Fichier '{file_type}' non disponible")

    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        media_type="application/octet-stream",
    )


@app.post("/api/v1/cancel/{job_id}", tags=["Jobs"])
def cancel_job(job_id: str):
    """Annulation gracieuse d'un job en cours."""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("queued", "processing"):
        return {"message": f"Job {job_id} already {job['status']}"}
    evt = cancel_events.get(job_id)
    if evt is None:
        evt = threading.Event()
        cancel_events[job_id] = evt
    evt.set()
    return {"message": f"Cancellation requested for {job_id}"}


@app.delete("/api/v1/job/{job_id}", tags=["Jobs"])
def cleanup_job(job_id: str):
    """Supprimer un job et son fichier audio uploadé."""
    with jobs_lock:
        job = jobs.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    audio_path = Path(job.get("audio_path", ""))
    if audio_path.exists():
        audio_path.unlink()
    return {"message": f"Job {job_id} cleaned up"}


@app.get("/api/v1/jobs", tags=["Jobs"])
def list_jobs(status: Optional[str] = None):
    """
    Lister tous les jobs, optionnellement filtrés par statut.

    status : queued | processing | completed | failed | cancelled
    """
    with jobs_lock:
        all_jobs = list(jobs.values())
    if status:
        all_jobs = [j for j in all_jobs if j["status"] == status]
    all_jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {
        "total": len(all_jobs),
        "jobs": [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "original_filename": j.get("original_filename"),
                "created_at": j["created_at"],
                "completed_at": j.get("completed_at"),
            }
            for j in all_jobs
        ],
    }


# --- Health ---

@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe — toujours 200 si le process API tourne."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health/services", tags=["Health"])
def health_services():
    """Vérifie la disponibilité des serveurs Whisper et LLM (~10s timeout)."""
    statuses = check_all_services(
        server_url=settings.server_url,
        llm_base_url=settings.llm_base_url,
        api_key=settings.api_key,
        whisper_model=settings.whisper_model,
        llm_model=settings.llm_model,
    )
    all_ok = all(s.ok for s in statuses)
    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": [asdict(s) for s in statuses],
    }
