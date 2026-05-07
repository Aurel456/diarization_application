# src/health_check.py
"""
Health check utilities for the Whisper transcription server and LLM server.
Checks model availability and response time.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI


@dataclass
class ServiceStatus:
    name: str          # "Whisper" or "LLM"
    url: str
    ok: bool
    model_found: bool  # True if the expected model is available
    available_models: List[str]
    latency_ms: Optional[int]  # round-trip latency in ms, None on error
    error: Optional[str]

    @property
    def label(self) -> str:
        if not self.ok:
            return f"❌ {self.name} — inaccessible ({self.error})"
        if not self.model_found:
            models_str = ", ".join(self.available_models[:5]) or "aucun"
            return f"⚠️ {self.name} — connecté mais modèle introuvable (dispo: {models_str})"
        return f"✅ {self.name} — OK ({self.latency_ms} ms)"


def check_service(
    name: str,
    base_url: str,
    api_key: Optional[str],
    expected_model: Optional[str],
    timeout: int = 10,
) -> ServiceStatus:
    """
    Check connectivity and model availability for an OpenAI-compatible endpoint.

    Args:
        name: Display name for the service (e.g. "Whisper", "LLM").
        base_url: The /v1 base URL of the service.
        api_key: API key (can be None / dummy).
        expected_model: Model name to look for in the model list.
        timeout: HTTP timeout in seconds.

    Returns:
        ServiceStatus with ok/error/latency information.
    """
    if not base_url:
        return ServiceStatus(
            name=name,
            url=base_url or "",
            ok=False,
            model_found=False,
            available_models=[],
            latency_ms=None,
            error="URL non configurée",
        )

    client = OpenAI(base_url=base_url, api_key=api_key or "dummy", timeout=timeout)

    t0 = time.monotonic()
    try:
        models_resp = client.models.list()
        latency_ms = int((time.monotonic() - t0) * 1000)
        available = [m.id for m in models_resp.data]
        model_found = (
            expected_model in available if expected_model else bool(available)
        )
        return ServiceStatus(
            name=name,
            url=base_url,
            ok=True,
            model_found=model_found,
            available_models=available,
            latency_ms=latency_ms,
            error=None,
        )
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        logging.warning("Health check failed for %s (%s): %s", name, base_url, exc)
        return ServiceStatus(
            name=name,
            url=base_url,
            ok=False,
            model_found=False,
            available_models=[],
            latency_ms=latency_ms,
            error=str(exc)[:120],
        )


def check_all_services(
    server_url: Optional[str],
    llm_base_url: Optional[str],
    api_key: Optional[str],
    whisper_model: Optional[str],
    llm_model: Optional[str],
    timeout: int = 10,
) -> List[ServiceStatus]:
    """
    Run health checks for the Whisper server and the LLM server.

    If both URLs are identical, the LLM check reuses the Whisper result
    (same server, same request) and only checks for the LLM model separately.

    Returns:
        List of ServiceStatus — one per distinct service.
    """
    results: List[ServiceStatus] = []

    whisper_status = check_service(
        name="Whisper",
        base_url=server_url or "",
        api_key=api_key,
        expected_model=whisper_model,
        timeout=timeout,
    )
    results.append(whisper_status)

    # If LLM is on the same server, reuse connectivity info but check LLM model
    effective_llm_url = llm_base_url or server_url or ""
    if effective_llm_url and effective_llm_url == (server_url or ""):
        if whisper_status.ok:
            model_found = llm_model in whisper_status.available_models if llm_model else bool(whisper_status.available_models)
            llm_status = ServiceStatus(
                name="LLM",
                url=effective_llm_url,
                ok=True,
                model_found=model_found,
                available_models=whisper_status.available_models,
                latency_ms=whisper_status.latency_ms,
                error=None,
            )
        else:
            llm_status = ServiceStatus(
                name="LLM",
                url=effective_llm_url,
                ok=False,
                model_found=False,
                available_models=[],
                latency_ms=None,
                error=whisper_status.error,
            )
    else:
        llm_status = check_service(
            name="LLM",
            base_url=effective_llm_url,
            api_key=api_key,
            expected_model=llm_model,
            timeout=timeout,
        )
    results.append(llm_status)

    return results
