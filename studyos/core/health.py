"""Pre-flight health diagnostics for local Ollama service and embedding models."""

from __future__ import annotations

from enum import Enum
import sys
import httpx
from pydantic import BaseModel, Field

from studyos.core.config import AppConfig, get_default_config


class HealthStatus(str, Enum):
    """Operational health status of the local Ollama service."""

    HEALTHY = "healthy"
    OFFLINE = "offline"
    MODEL_MISSING = "model_missing"
    ERROR = "error"


class OllamaHealthReport(BaseModel):
    """Structured diagnostic report for Ollama service availability."""

    status: HealthStatus
    base_url: str
    model_name: str
    message: str
    actionable_remediation: str | None = None
    installed_models: list[str] = Field(default_factory=list)
    version: str | None = None

    @property
    def is_ready(self) -> bool:
        """Return True if Ollama is running and the target model is installed."""
        return self.status == HealthStatus.HEALTHY


def check_ollama_health(
    config: AppConfig | None = None,
    timeout_seconds: float = 3.0,
    client: httpx.Client | None = None,
) -> OllamaHealthReport:
    """Check connectivity to Ollama daemon and verify embedding model presence."""
    cfg = config or get_default_config()
    base_url = cfg.ollama_base_url.rstrip("/")
    target_model = cfg.embedding_model

    should_close = client is None
    http_client = client or httpx.Client(
        timeout=timeout_seconds,
        trust_env=False,
        follow_redirects=False,
    )

    # 1. Attempt connection and tags query
    try:
        try:
            # Query version
            version: str | None = None
            try:
                ver_resp = http_client.get(f"{base_url}/api/version")
                if ver_resp.is_success:
                    version = ver_resp.json().get("version")
            except Exception:
                pass

            tags_resp = http_client.get(f"{base_url}/api/tags")
            if tags_resp.status_code != 200:
                return OllamaHealthReport(
                    status=HealthStatus.ERROR,
                    base_url=base_url,
                    model_name=target_model,
                    version=version,
                    message=f"Ollama returned HTTP status {tags_resp.status_code}",
                    actionable_remediation=f"Inspect Ollama service logs at {base_url}.",
                )

            data = tags_resp.json()
            models_list = data.get("models", [])
            installed = [m.get("name", "") for m in models_list if isinstance(m, dict)]

            # Check if target model matches any installed model
            target_clean = target_model.lower().split(":")[0]
            model_found = any(
                m.lower() == target_model.lower()
                or m.lower().startswith(f"{target_clean}:")
                or m.lower() == target_clean
                for m in installed
            )

            if not model_found:
                return OllamaHealthReport(
                    status=HealthStatus.MODEL_MISSING,
                    base_url=base_url,
                    model_name=target_model,
                    version=version,
                    installed_models=installed,
                    message=f"Ollama is running, but required embedding model '{target_model}' is not installed.",
                    actionable_remediation=(
                        f"Install the model locally using: 'ollama pull {target_model}'\n"
                        f"Currently available models: {', '.join(installed) if installed else 'None'}"
                    ),
                )

            return OllamaHealthReport(
                status=HealthStatus.HEALTHY,
                base_url=base_url,
                model_name=target_model,
                version=version,
                installed_models=installed,
                message=f"Ollama is healthy at {base_url} (model '{target_model}' is available).",
            )

        except (httpx.ConnectError, httpx.NetworkError):
            return OllamaHealthReport(
                status=HealthStatus.OFFLINE,
                base_url=base_url,
                model_name=target_model,
                message=f"Cannot connect to local Ollama daemon at {base_url}.",
                actionable_remediation=(
                    f"Ensure Ollama is running locally.\n"
                    f"1. Start the daemon with: 'ollama serve'\n"
                    f"2. Or launch the Ollama desktop application.\n"
                    f"3. Verify connectivity at {base_url}."
                ),
            )
        except httpx.TimeoutException:
            return OllamaHealthReport(
                status=HealthStatus.OFFLINE,
                base_url=base_url,
                model_name=target_model,
                message=f"Connection to Ollama daemon at {base_url} timed out after {timeout_seconds}s.",
                actionable_remediation=(
                    f"Ollama may be overloaded or unresponsive. Restart the Ollama daemon and retry."
                ),
            )
        except Exception as e:
            return OllamaHealthReport(
                status=HealthStatus.ERROR,
                base_url=base_url,
                model_name=target_model,
                message=f"Unexpected error while checking Ollama health: {e}",
                actionable_remediation="Inspect Ollama service configuration and connectivity.",
            )
    finally:
        if should_close:
            http_client.close()


def main() -> None:
    """CLI diagnostic entrypoint for pre-flight health checks."""
    report = check_ollama_health()
    print("=" * 60)
    print("StudyOS - Local AI Service Pre-flight Diagnostic")
    print("=" * 60)
    print(f"Status:       {report.status.value.upper()}")
    print(f"Base URL:     {report.base_url}")
    print(f"Model:        {report.model_name}")
    if report.version:
        print(f"Version:      {report.version}")
    print(f"Message:      {report.message}")
    if report.installed_models:
        print(f"Installed:    {', '.join(report.installed_models)}")
    if report.actionable_remediation:
        print("-" * 60)
        print("Action Required:")
        print(report.actionable_remediation)
    print("=" * 60)

    if not report.is_ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
