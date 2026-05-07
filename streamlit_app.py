from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv(".env")



from settings import AppEnv, settings
from main_pipeline import (
    PipelineConfig,
    PipelineError,
    PipelineResult,
    PIPELINE_STEP_LABELS,
    PIPELINE_STEP_SEQUENCE,
    run_pipeline,
)
from src.meeting_minutes import (
    MEETING_MINUTES_FORMATS,
    DEFAULT_FORMAT,
    generate_meeting_minutes,
    save_meeting_minutes,
    minutes_to_markdown,
)
from src.health_check import check_all_services, ServiceStatus

# ---------------------------------------------------------------------------
# WebRTC streaming support (optional — requires streamlit-webrtc + av)
# ---------------------------------------------------------------------------
_WEBRTC_AVAILABLE = False
_webrtc_streamer = None
_WebRtcMode = None

try:
    from streamlit_webrtc import (
        AudioProcessorBase as _WebRTCBase,
        webrtc_streamer as _webrtc_streamer,
        WebRtcMode as _WebRtcMode,
    )
    import av as _av
    import queue as _queue_module
    import numpy as _np_audio
    import io as _io_audio
    from pydub import AudioSegment as _AudioSegment

    class WhisperAudioProcessor(_WebRTCBase):
        """Buffers WebRTC audio frames and transcribes chunks via Whisper API."""

        def __init__(
            self,
            server_url: str,
            api_key: str,
            whisper_model: str,
            chunk_duration_s: int = 5,
        ) -> None:
            self._server_url = server_url
            self._api_key = api_key or "dummy"
            self._whisper_model = whisper_model
            self._chunk_duration_s = chunk_duration_s
            self._sound_chunk = _AudioSegment.empty()
            self._lock = threading.Lock()
            self._result_queue: "_queue_module.Queue" = _queue_module.Queue()
            self._is_processing = False

        def recv(self, frame: "_av.AudioFrame") -> "_av.AudioFrame":
            arr = frame.to_ndarray()
            # float planar (fltp) → int16
            if arr.dtype.kind == "f":
                arr = (arr * 32767).clip(-32768, 32767).astype(_np_audio.int16)
            # mix multichannel to mono
            if arr.ndim > 1:
                arr = arr.mean(axis=0).astype(_np_audio.int16)
            sound = _AudioSegment(
                data=arr.tobytes(),
                sample_width=2,
                frame_rate=frame.sample_rate,
                channels=1,
            )
            with self._lock:
                self._sound_chunk += sound
                if (
                    len(self._sound_chunk) >= self._chunk_duration_s * 1000
                    and not self._is_processing
                ):
                    chunk = self._sound_chunk
                    self._sound_chunk = _AudioSegment.empty()
                    self._is_processing = True
                    threading.Thread(
                        target=self._transcribe, args=(chunk,), daemon=True
                    ).start()
            return frame

        def _transcribe(self, sound: "_AudioSegment") -> None:
            try:
                sound = sound.set_frame_rate(16000).set_channels(1)
                buf = _io_audio.BytesIO()
                sound.export(buf, format="wav")
                buf.seek(0)
                from openai import OpenAI as _OpenAI

                client = _OpenAI(base_url=self._server_url, api_key=self._api_key)
                result = client.audio.transcriptions.create(
                    model=self._whisper_model,
                    file=("chunk.wav", buf, "audio/wav"),
                    language="fr",
                    prompt="Transcription précise en français.",
                )
                text = result.text.strip()
                if text:
                    self._result_queue.put(
                        {"timestamp": time.strftime("%H:%M:%S"), "text": text}
                    )
            except Exception as exc:
                self._result_queue.put(
                    {
                        "timestamp": time.strftime("%H:%M:%S"),
                        "text": f"[Erreur: {exc}]",
                    }
                )
            finally:
                with self._lock:
                    self._is_processing = False

    _WEBRTC_AVAILABLE = True

except Exception:
    pass

# Configuration de la page Streamlit
if settings.app_env == AppEnv.PRODUCTION:
    st.set_page_config(
        page_title="Audio Processing Pipeline",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
else:
    st.set_page_config(
        page_title="Audio Processing Pipeline",
        page_icon="🎙️",
        layout="wide",
    )

import dsfr
dsfr.apply()

class StreamlitLogHandler(logging.Handler):
    """Redirige les logs vers un composant texte Streamlit."""

    def __init__(
        self,
        placeholder: Optional[st.delta_generator.DeltaGenerator],
        max_messages: int = 1000,
    ):
        super().__init__()
        self.placeholder = placeholder
        self.messages: List[str] = []
        self.max_messages = max_messages

    def emit(self, record: logging.LogRecord) -> None:
        if self.placeholder is None:
            return
        try:
            msg = self.format(record)
            self.messages.append(msg)
            # Garder seulement les N derniers messages pour éviter de saturer l'UI
            if len(self.messages) > self.max_messages:
                self.messages = self.messages[-self.max_messages :]

            # Mise à jour du composant code block
            self.placeholder.code("\n".join(self.messages), language="text")
        except Exception:
            pass


def initialize_state() -> None:
    """Initialise les états de session Streamlit."""
    if "pipeline_running" not in st.session_state:
        st.session_state["pipeline_running"] = False
    if "step_status" not in st.session_state:
        st.session_state["step_status"] = {}
    if "pipeline_results" not in st.session_state:
        st.session_state["pipeline_results"] = None
    if "result_type" not in st.session_state:
        st.session_state["result_type"] = None
    if "batch_zip_path" not in st.session_state:
        st.session_state["batch_zip_path"] = None
    if "pipeline_error" not in st.session_state:
        st.session_state["pipeline_error"] = None
    if "pipeline_crash_logs" not in st.session_state:
        st.session_state["pipeline_crash_logs"] = []
    # Recording mode state
    if "recording_transcripts" not in st.session_state:
        st.session_state["recording_transcripts"] = []   # list of {"audio": bytes, "text": str}
    if "recording_running" not in st.session_state:
        st.session_state["recording_running"] = False
    # Direct pipeline launch from live recording
    if "recording_pipeline_trigger" not in st.session_state:
        st.session_state["recording_pipeline_trigger"] = False
    if "recording_pipeline_audio_path" not in st.session_state:
        st.session_state["recording_pipeline_audio_path"] = None
    # Cancellation event for in-flight pipelines
    if "pipeline_cancel_event" not in st.session_state:
        st.session_state["pipeline_cancel_event"] = None
    # Fine-grained transcription progress
    if "transcription_progress" not in st.session_state:
        st.session_state["transcription_progress"] = None
    # Health check cache
    if "health_status" not in st.session_state:
        st.session_state["health_status"] = None
    if "health_checked" not in st.session_state:
        st.session_state["health_checked"] = False


@contextmanager
def ephemeral_upload_dir(auto_delete: bool, grace_seconds: int) -> Path:
    """Gère un dossier temporaire pour les uploads avec nettoyage automatique."""
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_uploads_"))
    print(f"[STREAMLIT] Created temp upload dir: {temp_dir}")
    try:
        yield temp_dir
    finally:
        if auto_delete:

            def _cleanup() -> None:
                time.sleep(grace_seconds)
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        print(f"[STREAMLIT] Cleaned up temp dir: {temp_dir}")
                except Exception as e:
                    print(f"[STREAMLIT] Error cleaning temp dir: {e}")

            threading.Thread(target=_cleanup, daemon=True).start()


def schedule_path_cleanup(path: Path, grace_seconds: int) -> None:
    """Planifie la suppression d'un fichier ou dossier après un délai."""
    if grace_seconds <= 0:
        grace_seconds = 1

    def _cleanup() -> None:
        time.sleep(grace_seconds)
        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            print(f"[STREAMLIT] Scheduled cleanup executed for: {path}")
        except Exception as e:
            print(f"[STREAMLIT] Scheduled cleanup failed for {path}: {e}")

    threading.Thread(target=_cleanup, daemon=True).start()


def save_uploaded_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    destination_root: Path,
) -> List[str]:
    """Sauvegarde les fichiers uploadés sur le disque."""
    if not uploaded_files:
        return []
    destination_root.mkdir(parents=True, exist_ok=True)
    local_paths: List[str] = []
    for uploaded in uploaded_files:
        destination = destination_root / uploaded.name
        print(f"[STREAMLIT] Saving file: {uploaded.name} to {destination}")
        destination.write_bytes(uploaded.getbuffer())
        local_paths.append(str(destination.resolve()))
    return local_paths


def create_batch_zip(
    results: List[PipelineResult], output_dir: Path, zip_name: str = "Batch_Results.zip"
) -> Optional[str]:
    """Crée un fichier ZIP contenant tous les fichiers générés."""
    zip_path = output_dir / zip_name
    files_to_zip = []

    for res in results:
        # Ajout des fichiers existants au ZIP
        if res.docx_path and os.path.exists(res.docx_path):
            files_to_zip.append(res.docx_path)
        if res.txt_path and os.path.exists(res.txt_path):
            files_to_zip.append(res.txt_path)
        if res.srt_path and os.path.exists(res.srt_path):
            files_to_zip.append(res.srt_path)
        if res.summary_path and os.path.exists(
            res.summary_path
        ):  # Ajout du résumé au ZIP
            files_to_zip.append(res.summary_path)

    if not files_to_zip:
        print("[STREAMLIT] No files found to zip.")
        return None

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in files_to_zip:
                # On préserve le nom de fichier mais pas toute l'arborescence
                filename = os.path.basename(file)
                zipf.write(file, arcname=filename)
        print(f"[STREAMLIT] Batch zip created at: {zip_path}")
        return str(zip_path)
    except Exception as e:
        logging.error(f"Failed to create zip file: {e}")
        return None


from src.config_builder import build_pipeline_config as _build_pipeline_config_shared


def build_pipeline_config(
    audio_paths: List[str],
    settings_dict: Dict[str, any],
) -> PipelineConfig:
    """Thin wrapper around the shared config builder so sidebar values flow in."""
    overrides = dict(settings_dict)
    overrides.setdefault("log_to_console", False)
    overrides.setdefault("log_level", settings_dict.get("log_level", "INFO"))
    overrides["cancel_event"] = st.session_state.get("pipeline_cancel_event")
    return _build_pipeline_config_shared(audio_paths=audio_paths, overrides=overrides)


def run_health_check(server_url: str, llm_url: Optional[str], api_key: str, whisper_model: str, llm_model: Optional[str]) -> List[ServiceStatus]:
    """Run health checks and cache results in session state."""
    statuses = check_all_services(
        server_url=server_url,
        llm_base_url=llm_url,
        api_key=api_key,
        whisper_model=whisper_model,
        llm_model=llm_model,
        timeout=8,
    )
    st.session_state["health_status"] = statuses
    st.session_state["health_checked"] = True
    return statuses


def render_health_check_sidebar(server_url: str, llm_url: Optional[str], api_key: str, whisper_model: str, llm_model: Optional[str]) -> None:
    """Render health check status in the sidebar and show toasts on first check."""
    st.markdown("---")
    st.subheader("🔌 État des serveurs")

    col_btn, col_spin = st.columns([3, 1])
    with col_btn:
        do_check = st.button("Vérifier la connectivité", use_container_width=True)
    with col_spin:
        st.write("")

    if do_check or not st.session_state["health_checked"]:
        with st.spinner("Vérification..."):
            statuses = run_health_check(server_url, llm_url, api_key, whisper_model, llm_model)
        # Toasts (visible globally)
        for s in statuses:
            if s.ok and s.model_found:
                st.toast(f"✅ {s.name} opérationnel ({s.latency_ms} ms)", icon="✅")
            elif s.ok and not s.model_found:
                st.toast(f"⚠️ {s.name} connecté, modèle introuvable", icon="⚠️")
            else:
                st.toast(f"❌ {s.name} inaccessible : {s.error}", icon="❌")

    statuses = st.session_state.get("health_status") or []
    for s in statuses:
        if s.ok and s.model_found:
            st.success(s.label, icon=None)
        elif s.ok:
            st.warning(s.label, icon=None)
        else:
            st.error(s.label, icon=None)

        if s.ok and s.available_models:
            with st.expander(f"Modèles disponibles ({s.name})", expanded=False):
                for m in s.available_models:
                    st.code(m, language=None)


# ---------------------------------------------------------------------------
# Model discovery — populate selectboxes from `client.models.list()`
# ---------------------------------------------------------------------------

def fetch_available_models(server_url: Optional[str], api_key: Optional[str]) -> List[str]:
    """
    Query the OpenAI-compatible /models endpoint and return the list of model IDs.
    Returns [] on any failure so callers can fall back to a free-text input.
    """
    if not server_url:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(base_url=server_url, api_key=api_key or "dummy", timeout=8)
        return [m.id for m in client.models.list().data]
    except Exception as exc:
        logging.debug("Could not fetch model list from %s: %s", server_url, exc)
        return []


def get_cached_models(server_url: Optional[str], api_key: Optional[str]) -> List[str]:
    """
    Cached model list keyed by (server_url, api_key) in session state.
    Falls back to the health-check result if available; otherwise fetches once.
    """
    cache_key = ("__models_cache__", server_url, api_key)
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # Reuse health-check data if it covers the same URL
    statuses = st.session_state.get("health_status") or []
    for s in statuses:
        if s.url == server_url and s.available_models:
            st.session_state[cache_key] = list(s.available_models)
            return st.session_state[cache_key]

    models = fetch_available_models(server_url, api_key)
    st.session_state[cache_key] = models
    return models


def model_selectbox(
    label: str,
    current_value: Optional[str],
    available_models: List[str],
    key: str,
    help: Optional[str] = None,
) -> Optional[str]:
    """
    Render a model picker. Falls back to a free-text input when the server
    didn't return a model list (offline, error, etc.).

    The 'manual entry' option lets the user type a model name not advertised
    by the API.
    """
    if not available_models:
        return st.text_input(label, value=current_value or "", key=key, help=help) or None

    MANUAL = "✏️ Saisir manuellement…"
    options = list(available_models)
    if current_value and current_value not in options:
        options = [current_value] + options
    options = options + [MANUAL]

    default_idx = (
        options.index(current_value)
        if current_value and current_value in options
        else 0
    )
    choice = st.selectbox(label, options=options, index=default_idx, key=key, help=help)
    if choice == MANUAL:
        return st.text_input(
            f"{label} — saisie libre",
            value="",
            key=f"{key}_manual",
        ) or None
    return choice or None


def _transcribe_audio_bytes(audio_bytes: bytes, server_url: str, api_key: str, whisper_model: str) -> str:
    """Send raw audio bytes to the Whisper API and return the transcript text."""
    import io as _io
    import tempfile as _tmp
    from openai import OpenAI

    client = OpenAI(base_url=server_url, api_key=api_key or "dummy")
    with _tmp.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=whisper_model,
                file=audio_file,
                language="fr",
                prompt="Transcription précise en français.",
            )
        return result.text.strip()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def render_recording_section(server_url: str, api_key: str, whisper_model: str) -> None:
    """
    Live recording section:
    - st.audio_input() captures each utterance
    - Immediately transcribed via Whisper (streaming feel)
    - Transcript accumulates in session state
    - Button to export audio + run diarization pipeline
    """
    st.subheader("🎙️ Enregistrement et transcription en direct")
    st.caption(
        "Enregistrez plusieurs interventions successives. Chaque enregistrement est "
        "transcrit automatiquement. Lancez ensuite la diarization sur l'audio complet."
    )

    if not server_url or not api_key:
        st.warning("Configurez le Server URL et l'API Key dans la barre latérale pour utiliser l'enregistrement.")
        return

    col_rec, col_clear = st.columns([4, 1])
    with col_clear:
        if st.button("🗑️ Tout effacer", use_container_width=True):
            st.session_state["recording_transcripts"] = []
            st.rerun()

    with col_rec:
        audio_input = st.audio_input("Enregistrez votre intervention")

    if audio_input is not None:
        audio_bytes = audio_input.read()
        # Check if this recording is already in session (avoid double-processing on rerun)
        existing_hashes = {r.get("_hash") for r in st.session_state["recording_transcripts"]}
        audio_hash = hash(audio_bytes)

        if audio_hash not in existing_hashes:
            with st.spinner("Transcription en cours..."):
                try:
                    text = _transcribe_audio_bytes(audio_bytes, server_url, api_key, whisper_model)
                except Exception as exc:
                    text = f"[Erreur de transcription : {exc}]"

            st.session_state["recording_transcripts"].append({
                "_hash": audio_hash,
                "audio": audio_bytes,
                "text": text,
                "idx": len(st.session_state["recording_transcripts"]) + 1,
            })
            st.rerun()

    # Show accumulated transcript
    records = st.session_state["recording_transcripts"]
    if records:
        # Running metrics
        full_text = "\n".join(r["text"] for r in records)
        total_words = sum(len(r["text"].split()) for r in records)
        total_chars = sum(len(r["text"]) for r in records)

        m1, m2, m3 = st.columns(3)
        m1.metric("Interventions", len(records))
        m2.metric("Mots transcrits", total_words)
        m3.metric("Caractères", total_chars)

        st.markdown("**📝 Transcription en cours :**")
        for rec in records:
            st.markdown(f"**[{rec['idx']}]** {rec['text']}")

        st.divider()

        # Export full transcript
        col_dl, col_diarize = st.columns(2)
        with col_dl:
            st.download_button(
                "⬇️ Télécharger la transcription (.txt)",
                data=full_text.encode("utf-8"),
                file_name="transcription_live.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_diarize:
            if st.button(
                "🔬 Lancer la diarization sur cet audio",
                type="primary",
                use_container_width=True,
                help="Fusionne tous les enregistrements et lance immédiatement le pipeline complet.",
            ):
                from pydub import AudioSegment as _AS
                import io as _io

                combined = _AS.empty()
                for rec in records:
                    try:
                        seg = _AS.from_file(_io.BytesIO(rec["audio"]))
                        combined += seg
                    except Exception:
                        pass

                if len(combined) < 500:
                    st.error("L'audio combiné est trop court pour être diarisé.")
                else:
                    tmp_dir = Path(tempfile.mkdtemp(prefix="live_recording_"))
                    merged_path = tmp_dir / "live_recording_merged.wav"
                    combined.export(str(merged_path), format="wav")

                    # Trigger the full pipeline directly — no re-upload needed.
                    st.session_state["recording_pipeline_audio_path"] = str(merged_path.resolve())
                    st.session_state["recording_pipeline_trigger"] = True
                    st.toast(f"Audio fusionné ({len(combined)/1000:.1f}s) — lancement du pipeline…", icon="🚀")
                    st.rerun()
    else:
        st.info("Appuyez sur le bouton micro ci-dessus pour commencer l'enregistrement.")


def render_streaming_section(server_url: str, api_key: str, whisper_model: str) -> None:
    """
    Real-time streaming transcription using streamlit-webrtc.

    Captures audio continuously from the microphone, buffers it into chunks,
    and sends each chunk to Whisper for transcription. Results appear
    automatically as each chunk is processed.
    """
    st.subheader("🎙️ Transcription en temps réel (streaming)")
    st.caption(
        "Capture audio continu depuis le micro. Chaque chunk de quelques secondes "
        "est envoyé à Whisper pour transcription. Les résultats s'accumulent automatiquement."
    )

    if not _WEBRTC_AVAILABLE or _webrtc_streamer is None or _WebRtcMode is None:
        st.error(
            "**streamlit-webrtc** n'est pas disponible. "
            "Installez-le avec : `pip install streamlit-webrtc`"
        )
        return

    assert _webrtc_streamer is not None
    assert _WebRtcMode is not None

    if not server_url or not api_key:
        st.warning("Configurez le Server URL et l'API Key dans la barre latérale pour utiliser le streaming.")
        return

    if "streaming_transcripts" not in st.session_state:
        st.session_state["streaming_transcripts"] = []

    # Parameters + clear button
    col_p, col_c = st.columns([3, 1])
    with col_p:
        chunk_duration = st.slider(
            "Durée des chunks (secondes)",
            3, 15, 5, 1,
            key="stream_chunk_dur",
            help="Durée d'audio accumulée avant chaque envoi à Whisper.",
        )
    with col_c:
        st.write("")
        if st.button("🗑️ Effacer", use_container_width=True, key="clear_streaming"):
            st.session_state["streaming_transcripts"] = []
            st.rerun()

    # WebRTC streamer — START / STOP button rendered by the component itself
    ctx = _webrtc_streamer(
        key="whisper_stream",
        mode=_WebRtcMode.SENDONLY,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=lambda: WhisperAudioProcessor(
            server_url=server_url,
            api_key=api_key,
            whisper_model=whisper_model,
            chunk_duration_s=chunk_duration,
        ),
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
    )

    # Drain result queue from the processor into session state
    if ctx.audio_processor is not None:
        while True:
            try:
                result = ctx.audio_processor._result_queue.get_nowait()
                st.session_state["streaming_transcripts"].append(result)
            except Exception:
                break

    # Display accumulated transcripts
    st.divider()
    st.markdown("### 📝 Transcription")
    transcripts = st.session_state.get("streaming_transcripts", [])

    if transcripts:
        for entry in transcripts[-30:]:
            st.caption(f"**[{entry['timestamp']}]** {entry['text']}")

        total_chars = sum(len(t["text"]) for t in transcripts)
        st.caption(f"📊 {len(transcripts)} chunks · {total_chars} caractères")

        full_text = "\n".join(f"[{t['timestamp']}] {t['text']}" for t in transcripts)
        col_dl, _ = st.columns([1, 2])
        with col_dl:
            st.download_button(
                "⬇️ Télécharger la transcription (.txt)",
                data=full_text.encode("utf-8"),
                file_name="streaming_transcription.txt",
                mime="text/plain",
                use_container_width=True,
            )
    else:
        st.info("Cliquez sur **START** pour démarrer la capture audio.")

    # Auto-refresh while streaming is active (polls for new chunks)
    try:
        is_playing = ctx.state.playing
    except Exception:
        is_playing = False

    if is_playing:
        time.sleep(0.8)
        st.rerun()


def main() -> None:
    initialize_state()

    st.title("🎙️ Audio Processing Pipeline")

    with st.expander("ℹ️ Comment utiliser cette application", expanded=False):
        st.markdown(
            """
**Bienvenue !** Cette application transforme un fichier audio (réunion, interview, conférence…) en texte exploitable.

**3 étapes simples :**
1. **Choisissez la Tâche** ci-dessous selon votre besoin :
   - 🚀 **Transcription rapide** — convertit l'audio en texte (le plus rapide).
   - 📺 **Sous-titres (SRT)** — génère un fichier de sous-titres horodatés.
   - 🎤 **Détection des locuteurs + compte rendu** — sépare chaque intervenant et permet l'identification (manuelle ou par IA).
   - 🎙️ **Enregistrement Live** — enregistrez directement depuis le micro, puis lancez le pipeline complet.
   - 🎙️ **Streaming temps réel** — transcription continue pendant que vous parlez.

2. **Uploadez votre fichier** dans la barre latérale (formats supportés : `mp3`, `wav`, `m4a`, `flac`, `mp4`, `mkv`, `mov`).

3. **Activez les options** souhaitées (LLM cleaning, résumé, **compte rendu**, identification des locuteurs) et cliquez sur **🚀 Lancer le traitement**.

💡 **Astuce :** vous pouvez générer un **compte rendu professionnel** (CODIR, technique, RH, projet…) sur n'importe quelle tâche. Activez l'option « Générer un compte rendu » ci-dessous.
"""
        )

    # --- 1. SÉLECTION DU MODE DE TÂCHE ---
    col_mode, col_strategy = st.columns(2)

    with col_mode:
        task_mode = st.selectbox(
            "1. Choisissez la Tâche",
            options=[
                "Transcription rapide",
                "Sous-titres (SRT)",
                "Détection des locuteurs + compte rendu",
                "🎙️ Enregistrement Live",
                "🎙️ Streaming temps réel",
            ],
            help="Ce que vous voulez obtenir en sortie.",
        )

    # Flags logiques
    is_recording_mode = task_mode == "🎙️ Enregistrement Live"
    is_streaming_mode = task_mode == "🎙️ Streaming temps réel"
    is_simple_mode = task_mode in ["Transcription rapide", "Sous-titres (SRT)"]
    is_subtitle_mode = task_mode == "Sous-titres (SRT)"
    is_full_mode = task_mode == "Détection des locuteurs + compte rendu"

    # --- 2. SÉLECTION DE LA STRATÉGIE DE TRAITEMENT ---
    with col_strategy:
        strategy_options = ["📦 Batch (Indépendant)", "🔗 Fusionné (Séquentiel)"]
        if is_full_mode or is_recording_mode:
            strategy_options.append("⚡ Fusionné (Simultané)")

        processing_strategy = st.radio(
            "2. Stratégie de Traitement",
            options=strategy_options,
            help="Batch: fichiers sans rapport (ZIP). Séquentiel: fichiers suite à suite. Simultané: même moment.",
        )

    is_batch = "Batch" in processing_strategy
    is_sequential = "Séquentiel" in processing_strategy
    is_concurrent = "Simultané" in processing_strategy

    if is_subtitle_mode:
        st.info(
            "ℹ️ Mode Sous-titres : Granularité fine activée (word-level timestamps)."
        )

    if is_batch:
        st.caption("👉 Chaque fichier traité séparément ➜ ZIP.")
    elif is_sequential:
        st.caption(
            "👉 Fichiers considérés comme suite chronologique ➜ 1 fichier unique."
        )
    elif is_concurrent:
        st.caption(
            "👉 Fichiers traités en parallèle sur la même timeline ➜ 1 fichier unique."
        )

    # --- CENTRAL: Meeting minutes config (visible for ALL pipeline tasks) ---
    # Streaming-only / live-recording pre-pipeline modes don't run a pipeline,
    # so we hide the option there.
    central_meeting_minutes_enabled = False
    central_meeting_minutes_format = DEFAULT_FORMAT
    central_meeting_minutes_model: Optional[str] = None
    central_meeting_minutes_instructions: Optional[str] = None
    if not is_streaming_mode:
        with st.expander("📋 Compte rendu de réunion", expanded=False):
            st.caption(
                "Génère automatiquement un compte rendu professionnel à partir de la transcription. "
                "Disponible pour toutes les tâches (transcription rapide, sous-titres, diarization)."
            )
            central_meeting_minutes_enabled = st.checkbox(
                "Générer un compte rendu",
                value=settings.enable_meeting_minutes,
                key="central_mm_enable",
            )
            if central_meeting_minutes_enabled:
                mm_col1, mm_col2 = st.columns(2)
                with mm_col1:
                    central_meeting_minutes_format = st.selectbox(
                        "Format du compte rendu",
                        options=list(MEETING_MINUTES_FORMATS.keys()),
                        format_func=lambda k: MEETING_MINUTES_FORMATS[k]["label"],
                        key="central_mm_format",
                    )
                    st.caption(MEETING_MINUTES_FORMATS[central_meeting_minutes_format]["description"])
                with mm_col2:
                    central_meeting_minutes_model = model_selectbox(
                        "Modèle LLM",
                        current_value=settings.meeting_minutes_model or settings.llm_model,
                        available_models=get_cached_models(
                            settings.server_url, settings.api_key
                        ),
                        key="central_mm_model",
                    )
                central_meeting_minutes_instructions = st.text_area(
                    "Instructions spécifiques (optionnel)",
                    placeholder="Ex: Mets l'accent sur les décisions budgétaires. Utilise le vouvoiement.",
                    height=80,
                    key="central_mm_instructions",
                ) or None

    with st.sidebar:
        st.header("Audio Source")

        # --- BOUTON DE SECOURS ---
        if st.button("⚠️ Reset / Unblock State", type="secondary"):
            print("[STREAMLIT] Manual State Reset Triggered")
            st.session_state["pipeline_running"] = False
            st.session_state["pipeline_error"] = None
            st.session_state["pipeline_results"] = None
            st.rerun()

        uploaded_files = st.file_uploader(
            "Upload audio files",
            type=["mp3", "wav", "m4a", "flac", "mov", "mp4", "mkv"],
            accept_multiple_files=True,
        )

        # --- Paramètres ---
        is_production = settings.app_env == AppEnv.PRODUCTION

        run_id = None
        # Default chunks for simple mode
        chunk_size = 300
        segment_duration = settings.segment_duration
        min_speaker_duration = settings.min_speaker_duration
        max_workers = settings.max_workers
        vad_filter = settings.vad_filter
        reuse_cache = settings.default_reuse_cache
        force_split = settings.default_force_split
        clear_saved_state = settings.default_clear_saved_state
        enable_llm_cleaning = settings.enable_llm_cleaning
        enable_summary = settings.enable_summary
        # Single endpoint: LLM uses the same SERVER_URL as Whisper.
        llm_base_url = settings.server_url or None
        llm_model = settings.llm_model

        # Speaker count hints (None = auto / let HDBSCAN decide)
        num_speakers: Optional[int] = settings.num_speakers
        min_speakers: Optional[int] = settings.min_speakers
        max_speakers: Optional[int] = settings.max_speakers

        # Full-mode options — defaults override in the sidebar Full Mode block
        enable_speaker_identification = settings.enable_speaker_identification
        speaker_identification_model = settings.speaker_identification_model
        # Meeting minutes are now configured centrally (above the sidebar) and
        # apply to every task. Pull the values straight from those widgets.
        enable_meeting_minutes = central_meeting_minutes_enabled
        meeting_minutes_model = (
            central_meeting_minutes_model if central_meeting_minutes_enabled else None
        )
        meeting_minutes_format = (
            central_meeting_minutes_format
            if central_meeting_minutes_enabled
            else DEFAULT_FORMAT
        )
        meeting_minutes_instructions = (
            central_meeting_minutes_instructions
            if central_meeting_minutes_enabled
            else None
        )

        backend_processing_mode = "concurrent" if is_concurrent else "sequential"

        hf_token = settings.hf_token
        server_url = settings.server_url
        api_key = settings.api_key
        whisper_model = settings.whisper_model
        log_level = settings.default_log_level

        if is_production:
            st.header("Settings")
            st.info("Configuration managed by admin.")
        else:
            st.header("Settings")
            run_id = st.text_input("Run ID prefix (optional)")

            if is_simple_mode:
                if is_subtitle_mode:
                    st.caption("🔒 Chunk size: 300s (Optimisé pour SRT)")
                    chunk_size = 300
                else:
                    chunk_size = st.number_input("Chunk size (s)", 60, 3600, 600, 60)

                col_clean, col_sum = st.columns(2)
                with col_clean:
                    enable_llm_cleaning = st.checkbox(
                        "LLM Cleaning",
                        value=settings.enable_llm_cleaning,
                        help="Corrige orthographe, ponctuation, hésitations sans changer le fond.",
                    )
                with col_sum:
                    enable_summary = st.checkbox(
                        "Résumé",
                        value=settings.enable_summary,
                        help="Génère un résumé du contenu via LLM.",
                    )
                if enable_llm_cleaning or enable_summary:
                    llm_model = model_selectbox(
                        "LLM Model",
                        current_value=settings.llm_model,
                        available_models=get_cached_models(
                            settings.server_url, settings.api_key
                        ),
                        key="simple_llm_model",
                    )

            else:  # Full Mode
                segment_duration = st.number_input(
                    "Chunk duration (s)", 60, 3600, 1200, 60
                )
                min_speaker_duration = st.number_input(
                    "Min speaker (s)", 0.1, 5.0, 0.5, 0.1
                )
                max_workers = st.slider("Workers", 1, 8, 2)
                vad_filter = st.checkbox("VAD Filter", True)

                # --- Speaker count hints ---
                st.markdown("**🎯 Nombre de locuteurs**")
                _default_mode = (
                    "Connu exactement"
                    if settings.num_speakers
                    else "Plage min-max"
                    if (settings.min_speakers or settings.max_speakers)
                    else "Inconnu (auto)"
                )
                speaker_count_mode = st.radio(
                    "Hint pour le clustering",
                    ["Inconnu (auto)", "Connu exactement", "Plage min-max"],
                    index=["Inconnu (auto)", "Connu exactement", "Plage min-max"].index(_default_mode),
                    horizontal=True,
                    help=(
                        "Connu : force exactement N clusters via AgglomerativeClustering. "
                        "Plage : HDBSCAN puis re-clustering si en-dehors des bornes."
                    ),
                )
                if speaker_count_mode == "Connu exactement":
                    num_speakers = int(st.number_input(
                        "Nombre de locuteurs",
                        min_value=1, max_value=30,
                        value=int(settings.num_speakers or 2),
                        step=1,
                    ))
                    min_speakers = None
                    max_speakers = None
                elif speaker_count_mode == "Plage min-max":
                    _col_min, _col_max = st.columns(2)
                    with _col_min:
                        min_speakers = int(st.number_input(
                            "Min", min_value=1, max_value=30,
                            value=int(settings.min_speakers or 2), step=1,
                        ))
                    with _col_max:
                        max_speakers = int(st.number_input(
                            "Max", min_value=1, max_value=30,
                            value=int(settings.max_speakers or max(min_speakers, 5)),
                            step=1,
                        ))
                    if min_speakers > max_speakers:
                        st.warning(f"Min ({min_speakers}) > Max ({max_speakers}) — Max ajusté.")
                        max_speakers = min_speakers
                    num_speakers = None
                else:
                    num_speakers = None
                    min_speakers = None
                    max_speakers = None

                # --- LLM features (cleaning + summary, séparés) ---
                st.markdown("**🤖 Traitement LLM**")
                col_clean, col_sum = st.columns(2)
                with col_clean:
                    enable_llm_cleaning = st.checkbox(
                        "LLM Cleaning",
                        value=settings.enable_llm_cleaning,
                        help="Corrige orthographe, ponctuation, hésitations sans changer le fond.",
                    )
                with col_sum:
                    enable_summary = st.checkbox(
                        "Résumé",
                        value=settings.enable_summary,
                        help="Génère un résumé global du contenu via LLM.",
                    )
                if enable_llm_cleaning or enable_summary:
                    llm_model = model_selectbox(
                        "LLM Model",
                        current_value=settings.llm_model,
                        available_models=get_cached_models(
                            settings.server_url, settings.api_key
                        ),
                        key="full_llm_model",
                    )

                with st.expander("Advanced"):
                    reuse_cache = st.checkbox(
                        "Reuse cache", settings.default_reuse_cache
                    )
                    force_split = st.checkbox(
                        "Force split", settings.default_force_split
                    )
                    clear_saved_state = st.checkbox(
                        "Clear state", settings.default_clear_saved_state
                    )

                # Speaker Identification
                st.markdown("---")
                st.subheader("🔍 Identification des locuteurs")
                enable_speaker_identification = st.checkbox(
                    "Identifier les locuteurs (nom / rôle) via LLM",
                    value=settings.enable_speaker_identification,
                    help="Le LLM analyse la transcription complète pour déduire les noms et fonctions.",
                )
                if enable_speaker_identification:
                    speaker_identification_model = model_selectbox(
                        "Modèle Speaker ID",
                        current_value=settings.speaker_identification_model or settings.llm_model,
                        available_models=get_cached_models(
                            settings.server_url, settings.api_key
                        ),
                        key="speaker_id_model",
                    )
                else:
                    speaker_identification_model = None

                # Meeting Minutes — moved to the central interface (visible for all tasks).

            st.header("Credentials")
            hf_token_input = (
                st.text_input("HF Token", type="password") if not is_simple_mode else ""
            )
            hf_token = hf_token_input or settings.hf_token
            server_url = st.text_input("Server URL", value=settings.server_url)
            api_key = st.text_input("API Key", type="password", value=settings.api_key)
            whisper_model = model_selectbox(
                "Whisper Model",
                current_value=settings.whisper_model,
                available_models=get_cached_models(server_url, api_key),
                key="whisper_model",
            ) or "whisper"
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING"], index=0)

        # Health check (always shown, needs server_url/api_key)
        render_health_check_sidebar(
            server_url=server_url,
            llm_url=llm_base_url,
            api_key=api_key,
            whisper_model=whisper_model,
            llm_model=llm_model,
        )

    # --- MODE ENREGISTREMENT LIVE ---
    # If the user clicked "Lancer la diarization" in the recording section, we skip the recording UI
    # and let the pipeline run below using the merged recording as input.
    recording_triggered = st.session_state.get("recording_pipeline_trigger", False)
    if is_recording_mode and not recording_triggered:
        render_recording_section(server_url, api_key, whisper_model)
        return  # Don't show the pipeline UI below

    # Streaming mode — real-time transcription (no pipeline, just display)
    if is_streaming_mode:
        render_streaming_section(server_url, api_key, whisper_model)
        return  # Don't show the pipeline UI below

    # --- Zone de Feedback ---

    # 1. Zone d'erreur persistante (ne s'efface pas au rerun si définie)
    if st.session_state["pipeline_error"]:
        st.error(
            f"❌ Une erreur est survenue lors de l'exécution précédente :\n\n{st.session_state['pipeline_error']}"
        )
        if st.session_state["pipeline_crash_logs"]:
            with st.expander("🔍 Voir les logs de l'erreur"):
                st.code(
                    "\n".join(st.session_state["pipeline_crash_logs"][-30:]),
                    language="text",
                )

        if st.button("Effacer l'erreur"):
            st.session_state["pipeline_error"] = None
            st.session_state["pipeline_crash_logs"] = []
            st.rerun()

    # 2. Zone de progression active
    progress_placeholder = st.progress(0.0)
    status_placeholder = st.empty()

    # 3. Zone de Logs en temps réel
    log_placeholder = None
    if not settings.hide_log_panel:
        with st.expander(
            "📋 Logs d'exécution", expanded=(st.session_state["pipeline_running"])
        ):
            log_placeholder = st.empty()

    # Conteneur pour les résultats (affiché à la fin)
    result_container = st.container()

    # --- BOUTON D'EXÉCUTION ---
    # Désactivé si déjà en cours
    run_button = st.button(
        "🚀 Lancer le traitement",
        type="primary",
        use_container_width=True,
        disabled=st.session_state["pipeline_running"],
        help="Lancer traitement du fichier audio"
    )

    # Auto-trigger when a live recording has just been merged
    if recording_triggered and not st.session_state["pipeline_running"]:
        run_button = True
        st.info("🎙️ Lancement du pipeline sur l'enregistrement fusionné…")

    handler: Optional[StreamlitLogHandler] = None

    def progress_callback(
        step: str, status: str, message: str, payload: Optional[Dict] = None
    ) -> None:
        if step not in PIPELINE_STEP_SEQUENCE:
            return
        st.session_state["step_status"][step] = status
        completed_steps = [
            s
            for s in PIPELINE_STEP_SEQUENCE
            if st.session_state["step_status"].get(s) in {"success", "skipped"}
        ]
        progress = (
            len(completed_steps) / len(PIPELINE_STEP_SEQUENCE)
            if PIPELINE_STEP_SEQUENCE
            else 0
        )
        progress_placeholder.progress(progress)
        status_placeholder.markdown(
            f"**{PIPELINE_STEP_LABELS.get(step, step)}** : {message}",
            unsafe_allow_html=True,
        )

    # --- LOGIQUE D'EXÉCUTION ---
    if run_button and not st.session_state["pipeline_running"]:
        print("[STREAMLIT] Run button clicked. Initializing pipeline...")
        st.session_state["pipeline_running"] = True

        # Reset des états pour nouvelle run
        st.session_state["pipeline_error"] = None
        st.session_state["pipeline_results"] = None
        st.session_state["result_type"] = None
        st.session_state["batch_zip_path"] = None
        st.session_state["pipeline_crash_logs"] = []

        # Le bloc Try/Except principal capture TOUT pour éviter que l'UI ne crash sans message
        try:
            # Consume the recording trigger (so it fires only once)
            recording_audio_path: Optional[str] = None
            if recording_triggered:
                recording_audio_path = st.session_state.get("recording_pipeline_audio_path")
                st.session_state["recording_pipeline_trigger"] = False
                st.session_state["recording_pipeline_audio_path"] = None
                if not recording_audio_path or not os.path.exists(recording_audio_path):
                    raise ValueError("Enregistrement fusionné introuvable — relancez l'enregistrement.")
                # Force a full-pipeline run on the merged recording regardless of task_mode
                is_simple_mode_local = False
            else:
                is_simple_mode_local = is_simple_mode
                if not uploaded_files:
                    raise ValueError("Veuillez uploader au moins un fichier audio.")

            if not server_url or not api_key:
                raise ValueError("Identifiants manquants (URL Serveur ou Clé API).")

            # Création dossier temporaire pour l'upload (ou ré-utilisation du dossier de l'enregistrement)
            with ephemeral_upload_dir(
                settings.auto_delete_uploads, settings.cleanup_grace_seconds
            ) as upload_dir:
                if recording_audio_path:
                    status_placeholder.info("Préparation de l'enregistrement fusionné…")
                    dest = upload_dir / Path(recording_audio_path).name
                    shutil.copy(recording_audio_path, dest)
                    local_audio_paths = [str(dest.resolve())]
                else:
                    status_placeholder.info("Sauvegarde des fichiers uploadés...")
                    local_audio_paths = save_uploaded_files(uploaded_files, upload_dir)

                # Setup Logging vers l'interface
                if log_placeholder:
                    handler = StreamlitLogHandler(log_placeholder)
                    handler.setLevel(getattr(logging, log_level, logging.INFO))
                    logging.getLogger().addHandler(handler)

                st.session_state["step_status"] = {}

                # Préparation de la config de base
                # LLM URL/model are needed if ANY LLM-dependent feature is enabled
                _llm_used = enable_llm_cleaning or enable_summary or enable_speaker_identification or enable_meeting_minutes
                config_base = {
                    "server_url": server_url,
                    "api_key": api_key,
                    "whisper_model": whisper_model,
                    "log_level": log_level,
                    "simple_mode": is_simple_mode_local,
                    "enable_llm_cleaning": enable_llm_cleaning,
                    "enable_summary": enable_summary,
                    "llm_base_url": llm_base_url if _llm_used else None,
                    "llm_model": llm_model if _llm_used else None,
                    "hf_token": hf_token,
                }

                # Meeting minutes apply to every task (simple + full).
                config_base.update(
                    {
                        "enable_meeting_minutes": enable_meeting_minutes,
                        "meeting_minutes_model": meeting_minutes_model,
                        "meeting_minutes_format": meeting_minutes_format,
                        "meeting_minutes_instructions": meeting_minutes_instructions,
                    }
                )

                if is_simple_mode_local:
                    config_base["chunk_size"] = int(chunk_size)
                else:
                    config_base.update(
                        {
                            "segment_duration": int(segment_duration),
                            "min_speaker_duration": float(min_speaker_duration),
                            "max_workers": int(max_workers),
                            "vad_filter": vad_filter,
                            "reuse_cache": reuse_cache,
                            "force_split": force_split,
                            "clear_saved_state": clear_saved_state,
                            "processing_mode": backend_processing_mode,
                            "num_speakers": num_speakers,
                            "min_speakers": min_speakers,
                            "max_speakers": max_speakers,
                            "enable_speaker_identification": enable_speaker_identification,
                            "speaker_identification_model": speaker_identification_model,
                        }
                    )

                main_experiments_root = None

                # === ORCHESTRATION DU PIPELINE ===

                # CAS A: BATCH (Indépendant) OU Séquentiel Simple
                if is_batch or (is_simple_mode_local and is_sequential):
                    batch_results = []
                    total_files = len(local_audio_paths)
                    print(
                        f"[STREAMLIT] Mode Batch/Seq Simple selected. {total_files} files to process."
                    )

                    for idx, audio_path in enumerate(local_audio_paths):
                        filename = os.path.basename(audio_path)
                        status_placeholder.info(
                            f"Traitement fichier {idx + 1}/{total_files}: {filename}"
                        )
                        progress_placeholder.progress(0.0)  # Reset progress per file

                        current_run_id = f"{run_id or 'Run'}_{idx + 1}_{filename[:10]}"

                        file_config_dict = config_base.copy()
                        file_config_dict["run_id"] = current_run_id
                        # En mode batch simple, on traite comme séquentiel (1 fichier)
                        file_config_dict["processing_mode"] = "sequential"

                        config = build_pipeline_config([audio_path], file_config_dict)
                        if not main_experiments_root:
                            main_experiments_root = Path(config.experiments_root)

                        res = run_pipeline(
                            config,
                            progress_callback,
                            extra_log_handlers=[handler] if handler else None,
                        )
                        batch_results.append(res)

                    # Post-traitement des résultats
                    if is_simple_mode_local and is_sequential and not is_batch:
                        # Cas spécifique : Fusion manuelle des textes (SRT fusion non supportée facilement ici)
                        st.session_state["result_type"] = "merged_simple"

                        valid_results = [res for res in batch_results if res.success]
                        merged_text = "\n\n".join(
                            [
                                f"--- File: {os.path.basename(path)} ---\n{res.cleaned_text}"
                                for path, res in zip(local_audio_paths, batch_results)
                                if res.success
                            ]
                        )

                        st.session_state["pipeline_results"] = {
                            "text": merged_text,
                            "run_id": f"{run_id or 'Merged'}",
                            "count": len(valid_results),
                            # On garde les résultats individuels pour téléchargement aussi
                            "individual_results": valid_results,
                        }
                    else:
                        # Single-file Full Pipeline runs that landed here via Batch strategy
                        # (default selection in the UI) should display the rich single-result
                        # view (speaker identification, merge clusters, DOCX regen…).
                        # A 1-file "batch" with a docx_path is really a single Full Pipeline run.
                        if (
                            len(batch_results) == 1
                            and batch_results[0].success
                            and batch_results[0].docx_path
                        ):
                            st.session_state["result_type"] = "single"
                            st.session_state["pipeline_results"] = batch_results[0]
                            if settings.auto_delete_outputs and batch_results[0].experiments_dir:
                                schedule_path_cleanup(
                                    Path(batch_results[0].experiments_dir),
                                    settings.cleanup_grace_seconds,
                                )
                        else:
                            # Vrai Batch : on garde tout
                            st.session_state["result_type"] = "batch"
                            st.session_state["pipeline_results"] = batch_results
                            if main_experiments_root:
                                zip_path = create_batch_zip(
                                    batch_results, main_experiments_root
                                )
                                st.session_state["batch_zip_path"] = zip_path
                                if zip_path:
                                    schedule_path_cleanup(
                                        Path(zip_path), settings.cleanup_grace_seconds
                                    )

                # CAS B: FULL PIPELINE (Diarization) Fusionné — ou enregistrement live déclenché
                elif (is_full_mode or recording_audio_path) and (is_sequential or is_concurrent or recording_audio_path):
                    print(
                        f"[STREAMLIT] Full Pipeline Mode ({backend_processing_mode}) selected."
                    )
                    st.session_state["result_type"] = "single"
                    config_base["run_id"] = run_id
                    config = build_pipeline_config(local_audio_paths, config_base)

                    res = run_pipeline(
                        config,
                        progress_callback,
                        extra_log_handlers=[handler] if handler else None,
                    )
                    st.session_state["pipeline_results"] = res

                    if settings.auto_delete_outputs and res.experiments_dir:
                        schedule_path_cleanup(
                            Path(res.experiments_dir), settings.cleanup_grace_seconds
                        )

            # Fin du traitement réussie
            status_placeholder.success("Traitement terminé avec succès !")
            progress_placeholder.progress(1.0)

        except Exception as e:
            # Capture de l'erreur pour affichage persistant
            error_msg = str(e)
            print(f"[STREAMLIT] CRITICAL ERROR: {error_msg}")
            logging.error("Pipeline crash", exc_info=True)

            st.session_state["pipeline_error"] = error_msg
            # Sauvegarde des messages de logs courants pour analyse
            if handler:
                st.session_state["pipeline_crash_logs"] = handler.messages.copy()

        finally:
            st.session_state["pipeline_running"] = False
            if handler:
                logging.getLogger().removeHandler(handler)
            # On force le rerun pour mettre à jour l'interface (afficher erreur ou résultats)
            st.rerun()

    # --- AFFICHAGE RESULTATS (PERSISTANT) ---
    if st.session_state["pipeline_results"] is not None:
        r_type = st.session_state["result_type"]
        results = st.session_state["pipeline_results"]

        with result_container:
            st.divider()
            st.header("📂 Résultats")

            # 1. BATCH
            if r_type == "batch":
                st.success(f"✅ Batch traité ({len(results)} fichiers).")
                # If batch contains Full Pipeline (diarized) results, the rich per-file
                # UI (speaker identification, merge clusters, DOCX regen) is not shown
                # here. Hint the user to use a non-batch strategy if they need it.
                if any(getattr(r, "docx_path", None) for r in results):
                    st.info(
                        "ℹ️ Pour accéder à **🎧 Identification manuelle des locuteurs**, "
                        "**🔗 Fusion de clusters** et **📄 Regénération DOCX**, "
                        "relancez avec la stratégie **🔗 Fusionné (Séquentiel)** "
                        "ou **⚡ Fusionné (Simultané)** au lieu de Batch."
                    )
                zip_path = st.session_state.get("batch_zip_path")

                if zip_path and os.path.exists(zip_path):
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            "📦 Télécharger tout (ZIP)",
                            f,
                            "Batch_Results.zip",
                            "application/zip",
                            use_container_width=True,
                        )
                else:
                    st.warning("Le fichier ZIP n'a pas pu être généré.")

                with st.expander("Détails par fichier"):
                    for i, res in enumerate(results):
                        status_icon = "✅" if res.success else "❌"
                        st.write(f"**{i + 1}. {res.run_id}**: {status_icon}")
                        if res.summary_text:
                            st.caption(f"Résumé: {res.summary_text[:100]}...")

                # --- Compte rendu à la volée pour les résultats batch ---
                batch_cr_candidates = [
                    r for r in results
                    if r.success and not r.cleaned_data.empty
                ]
                if batch_cr_candidates:
                    st.divider()
                    st.subheader("📋 Générer des comptes rendus")
                    st.caption(
                        f"{len(batch_cr_candidates)} fichier(s) avec transcription disponible."
                    )

                    bcr_col1, bcr_col2 = st.columns(2)
                    with bcr_col1:
                        batch_cr_format = st.selectbox(
                            "Format du compte rendu",
                            options=list(MEETING_MINUTES_FORMATS.keys()),
                            format_func=lambda k: MEETING_MINUTES_FORMATS[k]["label"],
                            key="batch_cr_format",
                        )
                        st.caption(MEETING_MINUTES_FORMATS[batch_cr_format]["description"])
                    with bcr_col2:
                        batch_cr_instructions = st.text_area(
                            "Instructions spécifiques (optionnel)",
                            height=100,
                            key="batch_cr_instructions",
                        ) or None

                    if st.button(
                        "Générer les comptes rendus",
                        type="primary",
                        use_container_width=True,
                        key="batch_cr_button",
                        help="Générer les comptes rendus"
                    ):
                        llm_url = settings.llm_base_url or settings.server_url
                        llm_key = settings.api_key
                        llm_mod = settings.llm_model

                        if not llm_url or not llm_mod:
                            st.error("LLM_BASE_URL et LLM_MODEL doivent être configurés dans le .env.")
                        else:
                            import json as _json
                            for cr_res in batch_cr_candidates:
                                with st.spinner(f"Génération CR — {cr_res.run_id}..."):
                                    try:
                                        minutes = generate_meeting_minutes(
                                            transcript_df=cr_res.cleaned_data,
                                            llm_base_url=llm_url,
                                            llm_api_key=llm_key,
                                            llm_model=llm_mod,
                                            speaker_info={sid: v for sid, v in (cr_res.speaker_info or {}).items()} or None,
                                            format_key=batch_cr_format,
                                            user_instructions=batch_cr_instructions,
                                            timeout=180,
                                        )
                                    except Exception as _exc:
                                        minutes = None
                                        st.error(f"Erreur pour {cr_res.run_id}: {_exc}")

                                if minutes:
                                    md_content = minutes_to_markdown(minutes)
                                    dl_c1, dl_c2 = st.columns(2)
                                    dl_c1.download_button(
                                        f"⬇️ {cr_res.run_id} (.md)",
                                        data=md_content.encode("utf-8"),
                                        file_name=f"cr_{cr_res.run_id}_{batch_cr_format}.md",
                                        mime="text/markdown",
                                        use_container_width=True,
                                    )
                                    dl_c2.download_button(
                                        f"⬇️ {cr_res.run_id} (.json)",
                                        data=_json.dumps(minutes.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
                                        file_name=f"cr_{cr_res.run_id}_{batch_cr_format}.json",
                                        mime="application/json",
                                        use_container_width=True,
                                    )
                                    with st.expander(f"Aperçu — {cr_res.run_id}", expanded=False):
                                        st.markdown(md_content)
                                elif minutes is None:
                                    st.warning(f"Échec de génération pour {cr_res.run_id}. Consultez les logs.")

            # 2. MERGED SIMPLE (Mode séquentiel simple)
            elif r_type == "merged_simple":
                st.success(
                    f"✅ Transcription séquentielle terminée ({results['count']} fichiers)."
                )
                txt_data = results["text"]
                st.download_button(
                    "⬇️ Télécharger Transcription Fusionnée (TXT)",
                    txt_data,
                    f"{results['run_id']}_merged.txt",
                    "text/plain",
                    use_container_width=True,
                )

                st.info(
                    "💡 Note : En mode séquentiel simple, les fichiers SRT ne sont pas fusionnés automatiquement. Téléchargez les fichiers individuels ci-dessous si besoin."
                )

                # Option pour télécharger les fichiers individuels si besoin
                if "individual_results" in results:
                    with st.expander("Téléchargements individuels (SRT/TXT/Résumé)"):
                        for res in results["individual_results"]:
                            col_dl1, col_dl2, col_dl3 = st.columns(3)
                            if res.txt_path and os.path.exists(res.txt_path):
                                col_dl1.download_button(
                                    f"TXT - {Path(res.txt_path).name}",
                                    Path(res.txt_path).read_bytes(),
                                    Path(res.txt_path).name,
                                )
                            if res.srt_path and os.path.exists(res.srt_path):
                                col_dl2.download_button(
                                    f"SRT - {Path(res.srt_path).name}",
                                    Path(res.srt_path).read_bytes(),
                                    Path(res.srt_path).name,
                                )
                            if res.summary_path and os.path.exists(res.summary_path):
                                col_dl3.download_button(
                                    f"Résumé - {Path(res.summary_path).name}",
                                    Path(res.summary_path).read_bytes(),
                                    Path(res.summary_path).name,
                                )

            # 3. SINGLE RESULT (Pipeline simple ou Full unique)
            elif r_type == "single" or isinstance(results, PipelineResult):
                res = results

                # NOTE: Order matters — Full Pipeline ALSO produces a txt_path,
                # so we must check docx_path FIRST. Otherwise the simple branch
                # matches for Full runs and the speaker identification UI is
                # never rendered.
                _is_full_pipeline = bool(
                    res.docx_path and os.path.exists(res.docx_path)
                )

                # Mode SIMPLE / SRT (txt_path mais pas de DOCX)
                if not _is_full_pipeline and getattr(res, "txt_path", None):
                    # SRT is only relevant for the Sous-titres task, not for plain
                    # Transcription rapide. We hide that column when not subtitle mode.
                    show_srt = is_subtitle_mode
                    col1, col2, col3 = st.columns(3 if show_srt else 2)

                    # Bouton TXT
                    if res.txt_path and os.path.exists(res.txt_path):
                        col1.download_button(
                            "⬇️ Télécharger Texte (.txt)",
                            Path(res.txt_path).read_bytes(),
                            Path(res.txt_path).name,
                            "text/plain",
                            use_container_width=True,
                        )

                    # Bouton SRT (only in Sous-titres mode)
                    if show_srt:
                        if getattr(res, "srt_path", None) and os.path.exists(res.srt_path):
                            col2.download_button(
                                "⬇️ Télécharger Sous-titres (.srt)",
                                Path(res.srt_path).read_bytes(),
                                Path(res.srt_path).name,
                                "text/plain",
                                use_container_width=True,
                                type="primary",
                                help="Télécharger Sous-titres (.srt)"
                            )
                        else:
                            col2.warning("Fichier SRT non disponible.")

                    # Bouton Résumé — col2 in Transcription rapide, col3 in Sous-titres
                    summary_col = col3 if show_srt else col2
                    if getattr(res, "summary_path", None) and os.path.exists(
                        res.summary_path
                    ):
                        summary_col.download_button(
                            "⬇️ Télécharger Résumé (.txt)",
                            Path(res.summary_path).read_bytes(),
                            Path(res.summary_path).name,
                            "text/plain",
                            use_container_width=True,
                        )

                    # Compte rendu généré pendant le pipeline (mode rapide)
                    _mm_paths = res.meeting_minutes_paths or {}
                    if _mm_paths:
                        st.markdown("**📋 Compte rendu**")
                        _mm_cols = st.columns(min(len(_mm_paths), 2))
                        for _mm_i, (_mm_fmt, _mm_path) in enumerate(_mm_paths.items()):
                            if not os.path.exists(_mm_path):
                                continue
                            _mm_mime = (
                                "application/json"
                                if _mm_fmt == "json"
                                else "text/markdown"
                            )
                            _mm_cols[_mm_i % len(_mm_cols)].download_button(
                                f"⬇️ Compte rendu (.{_mm_fmt})",
                                data=Path(_mm_path).read_bytes(),
                                file_name=Path(_mm_path).name,
                                mime=_mm_mime,
                                use_container_width=True,
                                key=f"simple_mm_{_mm_fmt}",
                            )
                        if res.meeting_minutes:
                            with st.expander("Aperçu du compte rendu", expanded=False):
                                st.markdown(minutes_to_markdown(res.meeting_minutes))

                    # Aperçu
                    if res.cleaned_text:
                        with st.expander("Aperçu du texte"):
                            st.text_area("", res.cleaned_text, height=300)
                    if res.summary_text:
                        with st.expander("Aperçu du résumé"):
                            st.info(res.summary_text)

                # Mode FULL PIPELINE (Diarization)
                elif _is_full_pipeline:
                    st.metric("Interlocuteurs détectés", res.num_speakers_final or "N/A")

                    # --- Sélection des fichiers à télécharger ---
                    st.markdown("### 📁 Fichiers générés")

                    # Build the catalogue of available outputs
                    _output_catalogue: Dict[str, tuple] = {}
                    if res.docx_path and os.path.exists(res.docx_path):
                        _output_catalogue["docx"] = (
                            "📄 Rapport Word (.docx)",
                            res.docx_path,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    if res.txt_path and os.path.exists(res.txt_path):
                        _output_catalogue["txt"] = (
                            "📝 Texte brut (.txt)", res.txt_path, "text/plain"
                        )
                    if res.srt_path and os.path.exists(res.srt_path):
                        _output_catalogue["srt"] = (
                            "🔤 Sous-titres (.srt)", res.srt_path, "text/plain"
                        )
                    if getattr(res, "summary_path", None) and os.path.exists(res.summary_path):
                        _output_catalogue["summary"] = (
                            "📊 Résumé (.txt)", res.summary_path, "text/plain"
                        )
                    if res.speaker_identification_path and os.path.exists(
                        res.speaker_identification_path
                    ):
                        _output_catalogue["speaker_json"] = (
                            "👥 Identification locuteurs (.json)",
                            res.speaker_identification_path,
                            "application/json",
                        )
                    for _mm_fmt, _mm_path in (res.meeting_minutes_paths or {}).items():
                        if os.path.exists(_mm_path):
                            _mm_mime = (
                                "application/json"
                                if _mm_fmt == "json"
                                else "text/markdown"
                            )
                            _output_catalogue[f"cr_{_mm_fmt}"] = (
                                f"📋 Compte rendu ({_mm_fmt.upper()})",
                                _mm_path,
                                _mm_mime,
                            )

                    # Checkboxes — 3 per row
                    _keys = list(_output_catalogue.keys())
                    _n_cols = min(len(_keys), 3)
                    _selected: Dict[str, tuple] = {}
                    if _keys:
                        _chk_cols = st.columns(_n_cols)
                        for _i, _key in enumerate(_keys):
                            _label, _path, _mime = _output_catalogue[_key]
                            with _chk_cols[_i % _n_cols]:
                                if st.checkbox(
                                    _label,
                                    value=(_key == "docx"),
                                    key=f"dl_check_{_key}_{res.run_id}",
                                ):
                                    _selected[_key] = (_label, _path, _mime)

                    # Download buttons for selected files
                    if _selected:
                        _dl_cols = st.columns(len(_selected))
                        for _j, (_key, (_label, _path, _mime)) in enumerate(
                            _selected.items()
                        ):
                            with _dl_cols[_j]:
                                st.download_button(
                                    f"⬇️ {_label}",
                                    data=Path(_path).read_bytes(),
                                    file_name=Path(_path).name,
                                    mime=_mime,
                                    use_container_width=True,
                                    key=f"dl_btn_{_key}_{res.run_id}",
                                )

                    if res.summary_text:
                        with st.expander("Aperçu du résumé global"):
                            st.info(res.summary_text)

                    # =========================================================
                    # SECTION : VISUALISATION ET ÉTIQUETAGE DES LOCUTEURS
                    # =========================================================
                    # Always show this section when we have diarized data — audio clips
                    # are optional (require chunks_folder to still exist on disk).
                    if not res.cleaned_data.empty and (
                        "global_speaker" in res.cleaned_data.columns
                        or "speaker" in res.cleaned_data.columns
                    ):
                        st.divider()
                        st.subheader("🎧 Identification manuelle des locuteurs")
                        st.caption(
                            "Écoutez un extrait de chaque locuteur et saisissez son identité. "
                            "Ces informations seront utilisées pour générer le compte rendu."
                        )

                        # Pick speaker column up front — used by clip extraction and forms
                        speaker_col = (
                            "global_speaker" if "global_speaker" in res.cleaned_data.columns
                            else "speaker"
                        )
                        all_speakers = sorted(res.cleaned_data[speaker_col].dropna().unique())

                        # Initialiser le cache des clips audio (graceful when folder missing)
                        chunks_folder_ok = bool(
                            res.chunks_folder and os.path.isdir(res.chunks_folder)
                        )
                        if "speaker_audio_clips" not in st.session_state:
                            if chunks_folder_ok and res.chunks_folder:
                                with st.spinner("Extraction des extraits audio..."):
                                    try:
                                        from src.speaker_audio_sampler import extract_speaker_samples
                                        clips = extract_speaker_samples(
                                            res.cleaned_data,
                                            res.chunks_folder,
                                            res.segment_duration,
                                            n_samples=2,
                                            max_clip_duration_s=15.0,
                                        )
                                        st.session_state["speaker_audio_clips"] = clips
                                    except Exception as _e:
                                        st.session_state["speaker_audio_clips"] = {}
                                        st.warning(f"Impossible d'extraire les extraits audio : {_e}")
                            else:
                                st.session_state["speaker_audio_clips"] = {}
                                st.info(
                                    "ℹ️ Les extraits audio ne sont pas disponibles "
                                    f"(dossier chunks introuvable : `{res.chunks_folder or '—'}`). "
                                    "Vous pouvez tout de même saisir les noms et fusionner les clusters."
                                )

                        # clips is now Dict[str, List[bytes]] (possibly empty)
                        clips = st.session_state.get("speaker_audio_clips", {})

                        # Récupérer les labels LLM déjà identifiés (s'il y en a)
                        llm_labels: Dict[str, Dict] = res.speaker_info or {}

                        # Initialiser les labels manuels — itère sur all_speakers (pas clips)
                        # pour que le formulaire s'affiche même sans extraits audio.
                        speaker_ids_for_form = [
                            str(sid) for sid in all_speakers if str(sid).lower() != "noise"
                        ]
                        if "manual_speaker_labels" not in st.session_state:
                            st.session_state["manual_speaker_labels"] = {
                                sid: {
                                    "prenom": llm_labels.get(sid, {}).get("prenom", ""),
                                    "nom": llm_labels.get(sid, {}).get("nom", ""),
                                    "fonction": llm_labels.get(sid, {}).get("fonction", ""),
                                }
                                for sid in speaker_ids_for_form
                            }
                        text_col = (
                            "cleaned_transcription" if "cleaned_transcription" in res.cleaned_data.columns
                            else "transcription"
                        )

                        for sid in all_speakers:
                            sid = str(sid)
                            if sid.lower() == "noise":
                                continue

                            with st.expander(f"🎤 {sid}", expanded=True):
                                col_audio, col_form = st.columns([1, 1])

                                with col_audio:
                                    speaker_clips = clips.get(sid, [])
                                    if speaker_clips:
                                        for clip_i, clip_bytes in enumerate(speaker_clips, 1):
                                            st.caption(f"Extrait {clip_i}")
                                            st.audio(clip_bytes, format="audio/wav")
                                    else:
                                        st.info("Aucun extrait disponible")

                                    # Aperçu transcription
                                    sample_rows = res.cleaned_data[
                                        res.cleaned_data[speaker_col] == sid
                                    ].head(3)
                                    for _, row in sample_rows.iterrows():
                                        t = str(row.get(text_col, "")).strip()
                                        if t:
                                            st.caption(f'"{t[:120]}{"…" if len(t) > 120 else ""}"')

                                with col_form:
                                    current = st.session_state["manual_speaker_labels"].get(
                                        sid, {"prenom": "", "nom": "", "fonction": ""}
                                    )
                                    prenom = st.text_input(
                                        "Prénom", value=current["prenom"], key=f"prenom_{sid}"
                                    )
                                    nom = st.text_input(
                                        "Nom", value=current["nom"], key=f"nom_{sid}"
                                    )
                                    fonction = st.text_input(
                                        "Fonction / Service", value=current["fonction"],
                                        key=f"fonction_{sid}"
                                    )
                                    st.session_state["manual_speaker_labels"][sid] = {
                                        "prenom": prenom,
                                        "nom": nom,
                                        "fonction": fonction,
                                    }

                        # Boutons d'action sur les labels manuels
                        manual_labels = st.session_state.get("manual_speaker_labels", {})
                        has_any_manual = any(
                            v["prenom"] or v["nom"] for v in manual_labels.values()
                        )
                        btn_cols = st.columns(2)
                        with btn_cols[0]:
                            if has_any_manual:
                                import json as _json
                                st.download_button(
                                    "⬇️ Exporter les labels (JSON)",
                                    data=_json.dumps(manual_labels, ensure_ascii=False, indent=2),
                                    file_name="speaker_labels_manual.json",
                                    mime="application/json",
                                    use_container_width=True,
                                )
                        with btn_cols[1]:
                            if st.button(
                                "♻️ Réinitialiser depuis LLM",
                                use_container_width=True,
                                help="Remet à zéro les formulaires avec les suggestions du LLM (ou vide si aucune).",
                                key="reset_manual_labels",
                            ):
                                st.session_state["manual_speaker_labels"] = {
                                    sid: {
                                        "prenom": llm_labels.get(sid, {}).get("prenom", ""),
                                        "nom": llm_labels.get(sid, {}).get("nom", ""),
                                        "fonction": llm_labels.get(sid, {}).get("fonction", ""),
                                    }
                                    for sid in speaker_ids_for_form
                                }
                                # Clear widget state so defaults take effect
                                for sid in speaker_ids_for_form:
                                    for field in ("prenom", "nom", "fonction"):
                                        st.session_state.pop(f"{field}_{sid}", None)
                                st.rerun()

                        # ------------------------------------------------------
                        # Merge / rename clusters
                        # ------------------------------------------------------
                        st.markdown("---")
                        st.markdown("**🔗 Fusionner des clusters** — cocher ≥ 2 locuteurs à fusionner sous un identifiant cible.")

                        all_ids = sorted([str(s) for s in all_speakers if str(s).lower() != "noise"])
                        if "cluster_merge_selection" not in st.session_state:
                            st.session_state["cluster_merge_selection"] = []

                        merge_cols = st.columns([2, 1, 1])
                        with merge_cols[0]:
                            selected = st.multiselect(
                                "Clusters à fusionner",
                                options=all_ids,
                                default=st.session_state["cluster_merge_selection"],
                                key="cluster_merge_multiselect",
                            )
                            st.session_state["cluster_merge_selection"] = selected
                        with merge_cols[1]:
                            target = st.selectbox(
                                "Cible",
                                options=all_ids,
                                index=0 if all_ids else None,
                                key="cluster_merge_target",
                                help="Les locuteurs sélectionnés seront renommés en ce locuteur.",
                            )
                        with merge_cols[2]:
                            st.write("")
                            st.write("")
                            do_merge = st.button(
                                "Appliquer la fusion", type="primary",
                                disabled=(len(selected) < 2 or not target),
                                use_container_width=True,
                            )

                        if do_merge and target and len(selected) >= 2:
                            from src.exporter import apply_speaker_mapping
                            mapping = {s: target for s in selected if s != target}
                            st.session_state["cluster_merge_override"] = {
                                **st.session_state.get("cluster_merge_override", {}),
                                **mapping,
                            }
                            st.toast(f"{len(mapping)} cluster(s) fusionné(s) vers {target}", icon="🔗")
                            st.rerun()

                        active_overrides = st.session_state.get("cluster_merge_override", {})
                        if active_overrides:
                            st.caption(
                                "Fusions actives : " + ", ".join(
                                    f"{src}→{dst}" for src, dst in active_overrides.items()
                                )
                            )
                            if st.button("🗑️ Réinitialiser les fusions", key="reset_merges"):
                                st.session_state["cluster_merge_override"] = {}
                                st.rerun()

                        # ------------------------------------------------------
                        # Re-export DOCX with manual labels / merged clusters
                        # ------------------------------------------------------
                        if has_any_manual or active_overrides:
                            st.markdown("---")
                            if st.button("📄 Regénérer le DOCX avec labels + fusions", type="secondary", use_container_width=True, key="regen_docx"):
                                from src.exporter import re_export_docx_with_labels, apply_speaker_mapping
                                import tempfile as _tmp

                                with st.spinner("Regénération du DOCX..."):
                                    source_df = res.cleaned_data
                                    if active_overrides:
                                        source_df = apply_speaker_mapping(source_df, active_overrides)

                                    # Build speaker_info from manual labels, remapped through overrides
                                    effective_speaker_info = {}
                                    for sid_raw, labels in manual_labels.items():
                                        final_id = active_overrides.get(sid_raw, sid_raw)
                                        if labels.get("prenom") or labels.get("nom"):
                                            effective_speaker_info[final_id] = labels

                                    out_dir = _tmp.mkdtemp(prefix="docx_regen_")
                                    out_path = os.path.join(out_dir, "Transcription_Final_manual.docx")
                                    result_path = re_export_docx_with_labels(
                                        source_df, out_path,
                                        speaker_info=effective_speaker_info or None,
                                        audio_label=res.run_id,
                                    )
                                if result_path and os.path.exists(result_path):
                                    st.success("DOCX regénéré.")
                                    with open(result_path, "rb") as f:
                                        st.download_button(
                                            "⬇️ Télécharger le DOCX regénéré",
                                            data=f.read(),
                                            file_name="Transcription_Final_manual.docx",
                                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            use_container_width=True,
                                        )
                                else:
                                    st.error("La regénération a échoué.")

                    # =========================================================
                    # SECTION : GÉNÉRATION DE COMPTE RENDU À LA VOLÉE
                    # =========================================================
                    if not res.cleaned_data.empty:
                        st.divider()
                        st.subheader("📋 Générer un compte rendu")

                        # Source des labels locuteurs pour le CR
                        manual_labels = st.session_state.get("manual_speaker_labels", {})
                        has_manual = any(v.get("prenom") or v.get("nom") for v in manual_labels.values())
                        has_llm = bool(res.speaker_info)

                        speaker_source_options = ["Sans identification (anonyme)"]
                        if has_llm:
                            speaker_source_options.append("Identification LLM automatique")
                        if has_manual:
                            speaker_source_options.append("Labels manuels (saisis ci-dessus)")

                        speaker_source = st.radio(
                            "Utiliser les noms de locuteurs :",
                            options=speaker_source_options,
                            horizontal=True,
                        )

                        cr_col1, cr_col2 = st.columns(2)
                        with cr_col1:
                            cr_format = st.selectbox(
                                "Format du compte rendu",
                                options=list(MEETING_MINUTES_FORMATS.keys()),
                                format_func=lambda k: MEETING_MINUTES_FORMATS[k]["label"],
                                key="cr_format_adhoc",
                            )
                            st.caption(MEETING_MINUTES_FORMATS[cr_format]["description"])

                        with cr_col2:
                            cr_context_titre = st.text_input("Titre de la réunion (optionnel)", key="cr_titre")
                            cr_context_date = st.text_input("Date (optionnel)", placeholder="JJ/MM/AAAA", key="cr_date")

                        cr_instructions = st.text_area(
                            "Instructions spécifiques (optionnel)",
                            placeholder="Ex: Concentre-toi sur les décisions budgétaires. Utilise le vouvoiement.",
                            height=80,
                            key="cr_instructions",
                        ) or None

                        if st.button("Générer le compte rendu", type="primary", use_container_width=True):
                            # Résoudre les labels à utiliser
                            if "Labels manuels" in speaker_source and has_manual:
                                cr_speaker_info = {
                                    sid: {"prenom": v["prenom"], "nom": v["nom"], "fonction": v["fonction"]}
                                    for sid, v in manual_labels.items()
                                    if v.get("prenom") or v.get("nom")
                                }
                            elif "LLM" in speaker_source and has_llm:
                                cr_speaker_info = res.speaker_info
                            else:
                                cr_speaker_info = None

                            cr_context = {}
                            if cr_context_titre:
                                cr_context["titre"] = cr_context_titre
                            if cr_context_date:
                                cr_context["date"] = cr_context_date

                            llm_url = settings.llm_base_url or settings.server_url
                            llm_key = settings.api_key
                            llm_mod = settings.llm_model

                            if not llm_url or not llm_mod:
                                st.error("LLM_BASE_URL et LLM_MODEL doivent être configurés.")
                            else:
                                # Apply cluster merges to the transcript passed to the LLM
                                active_overrides = st.session_state.get("cluster_merge_override", {})
                                source_df = res.cleaned_data
                                if active_overrides:
                                    from src.exporter import apply_speaker_mapping
                                    source_df = apply_speaker_mapping(source_df, active_overrides)

                                with st.spinner(f"Génération du compte rendu ({MEETING_MINUTES_FORMATS[cr_format]['label']})..."):
                                    minutes = generate_meeting_minutes(
                                        transcript_df=source_df,
                                        llm_base_url=llm_url,
                                        llm_api_key=llm_key,
                                        llm_model=llm_mod,
                                        speaker_info=cr_speaker_info,
                                        meeting_context=cr_context or None,
                                        format_key=cr_format,
                                        user_instructions=cr_instructions,
                                        timeout=180,
                                    )

                                if minutes:
                                    st.success("Compte rendu généré !")
                                    md_content = minutes_to_markdown(minutes)

                                    dl_col1, dl_col2 = st.columns(2)
                                    dl_col1.download_button(
                                        "⬇️ Télécharger Markdown (.md)",
                                        data=md_content.encode("utf-8"),
                                        file_name=f"compte_rendu_{cr_format}.md",
                                        mime="text/markdown",
                                        use_container_width=True,
                                    )
                                    import json as _json
                                    dl_col2.download_button(
                                        "⬇️ Télécharger JSON (.json)",
                                        data=_json.dumps(minutes.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
                                        file_name=f"compte_rendu_{cr_format}.json",
                                        mime="application/json",
                                        use_container_width=True,
                                    )

                                    with st.expander("Aperçu du compte rendu", expanded=True):
                                        st.markdown(md_content)
                                else:
                                    st.error("La génération du compte rendu a échoué. Consultez les logs.")

                    # --- Aperçu des segments (développable) ---
                    if not res.cleaned_data.empty:
                        with st.expander("Aperçu des segments"):
                            cols_seg = ["start", speaker_col if "global_speaker" in res.cleaned_data.columns else "speaker", "cleaned_transcription" if "cleaned_transcription" in res.cleaned_data.columns else "transcription"]
                            available = [c for c in cols_seg if c in res.cleaned_data.columns]
                            st.dataframe(res.cleaned_data[available].head(30))


if __name__ == "__main__":
    main()
