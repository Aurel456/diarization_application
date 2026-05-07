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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from main_pipeline import (
    run_pipeline,
    PipelineConfig,
    PipelineResult,
)
from settings import settings

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


# Models
class TranscriptionRequest(BaseModel):
    """Request model for transcription (when providing file path instead of upload)."""

    audio_path: str
    run_id: Optional[str] = None
    enable_speaker_identification: Optional[bool] = None
    enable_meeting_minutes: Optional[bool] = None


class JobStatus(BaseModel):
    """Response model for job status."""

    job_id: str
    status: str  # "processing", "completed", "failed"
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None


# Background task function
def process_audio_job(job_id: str, audio_path: str, config_overrides: Dict[str, Any]):
    """
    Background task to run the audio processing pipeline.

    Args:
        job_id: Unique job identifier
        audio_path: Path to the audio file
        config_overrides: Override values for PipelineConfig
    """
    try:
        with jobs_lock:
            jobs[job_id]["status"] = "processing"

        logger.info(f"Starting job {job_id} for audio: {audio_path}")

        # Build pipeline config via the shared builder
        from src.config_builder import build_pipeline_config as _build

        overrides = {k: v for k, v in config_overrides.items() if v is not None}
        overrides["run_id"] = overrides.get("run_id") or f"api_{job_id}"
        # Attach the cancellation event for this job
        overrides["cancel_event"] = cancel_events.setdefault(job_id, threading.Event())

        base_config = _build(audio_paths=[audio_path], overrides=overrides)

        # Run pipeline
        result = run_pipeline(base_config)

        # Serialize result (exclude large objects)
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
            jobs[job_id].update(
                {
                    "status": "completed",
                    "result": result_dict,
                    "completed_at": datetime.utcnow().isoformat(),
                }
            )

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        # Map specific error classes to distinct statuses so clients can branch
        from src.errors import CancelledError, PipelineError
        if isinstance(e, CancelledError):
            final_status = "cancelled"
        elif isinstance(e, PipelineError):
            final_status = "failed"
        else:
            final_status = "failed"
        with jobs_lock:
            jobs[job_id].update(
                {
                    "status": final_status,
                    "message": str(e),
                    "completed_at": datetime.utcnow().isoformat(),
                }
            )
    finally:
        # Free the event entry
        cancel_events.pop(job_id, None)
        # In stateless mode, immediately delete the uploaded file once we're done
        # — the pipeline already copied it into experiments_dir if needed.
        if settings.stateless:
            try:
                up = Path(audio_path)
                if up.exists():
                    up.unlink(missing_ok=True)
                    logger.info(f"Stateless cleanup: removed upload {up}")
            except Exception as cleanup_exc:
                logger.warning(f"Failed to remove upload {audio_path}: {cleanup_exc}")


# Endpoints
@app.get("/")
def root():
    return {"message": "Audio Processing API", "version": "1.0.0"}


@app.post("/api/v1/transcribe", response_model=JobStatus)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    run_id: Optional[str] = None,
    enable_speaker_identification: Optional[bool] = None,
    enable_meeting_minutes: Optional[bool] = None,
):
    """
    Upload an audio file and start transcription/diarization pipeline.

    Returns a job ID that can be used to check status and download results.
    """
    # Validate file type
    allowed_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi", ".mkv"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}",
        )

    # Save uploaded file. In stateless mode (Docker) put it under the system
    # temp dir so it gets wiped at container restart; otherwise stay under
    # ./api_uploads/ for easier debugging on a dev machine.
    job_id = str(uuid.uuid4())
    if settings.stateless:
        upload_dir = Path(tempfile.gettempdir()) / "diarization_api_uploads"
    else:
        upload_dir = Path("api_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    audio_path = upload_dir / f"{job_id}{file_ext}"

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Register job
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "audio_path": str(audio_path),
            "original_filename": file.filename,
        }

    # Queue background task
    config_overrides = {
        "run_id": run_id,
        "enable_speaker_identification": enable_speaker_identification,
        "enable_meeting_minutes": enable_meeting_minutes,
    }

    background_tasks.add_task(
        process_audio_job, job_id, str(audio_path), config_overrides
    )

    return JobStatus(
        job_id=job_id,
        status="queued",
        created_at=jobs[job_id]["created_at"],
    )


@app.get("/api/v1/status/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get the status of a processing job."""
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


@app.get("/api/v1/download/{job_id}/{file_type}")
def download_result(job_id: str, file_type: str):
    """
    Download a result file for a completed job.

    file_type options: docx, txt, srt, summary, speaker_json, meeting_json, meeting_md
    """
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    result = job.get("result", {})
    file_path = None

    mapping = {
        "docx": result.get("docx_path"),
        "txt": result.get("txt_path"),
        "srt": result.get("srt_path"),
        "summary": result.get("summary_path"),
        "speaker_json": result.get("speaker_identification_path"),
    }

    # Meeting minutes have multiple files
    meeting_paths = result.get("meeting_minutes_paths", {})
    if file_type in meeting_paths:
        file_path = meeting_paths[file_type]
    else:
        file_path = mapping.get(file_type)

    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404, detail=f"File type '{file_type}' not available"
        )

    filename = Path(file_path).name
    return FileResponse(
        path=file_path, filename=filename, media_type="application/octet-stream"
    )


@app.post("/api/v1/cancel/{job_id}")
def cancel_job(job_id: str):
    """Request graceful cancellation of a running job."""
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


@app.delete("/api/v1/job/{job_id}")
def cleanup_job(job_id: str):
    """Delete a job and its associated files (for cleanup)."""
    with jobs_lock:
        job = jobs.pop(job_id, None)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Optionally delete uploaded file and generated outputs
    audio_path = Path(job.get("audio_path", ""))
    if audio_path.exists():
        audio_path.unlink()

    # Could also clean up experiment directories if needed

    return {"message": f"Job {job_id} cleaned up"}


# Health check
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
