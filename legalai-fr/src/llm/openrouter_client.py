from __future__ import annotations

import os
import json
import urllib.error
import urllib.request
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/mistral-small-3.2-24b-instruct"
DEFAULT_TIMEOUT_SECONDS = 60


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY est manquant. Ajoute la cle dans un fichier .env "
                "ou dans les variables d'environnement."
            )

    def summarize_legal_text(
        self,
        text: str,
        max_tokens: int = 700,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Le texte fourni est vide.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant juridique francais. Resume les documents "
                        "juridiques de facon fiable, structuree et prudente. Ne donne "
                        "pas de conseil juridique personnalise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Resume ce texte juridique en francais avec les sections "
                        "suivantes : faits, procedure, question juridique, decision, "
                        "points importants.\n\n"
                        f"Texte :\n{text}"
                    ),
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://127.0.0.1:8000"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "LegalAI-FR"),
        }

        request = urllib.request.Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise OpenRouterError(
                f"Erreur OpenRouter HTTP {exc.code} : {error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(
                f"Erreur reseau OpenRouter : {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise OpenRouterError(
                f"OpenRouter n'a pas repondu apres {self.timeout_seconds} secondes."
            ) from exc
        except json.JSONDecodeError as exc:
            raise OpenRouterError(
                "OpenRouter a retourne une reponse JSON invalide."
            ) from exc

        choices = data.get("choices") or []

        if not choices:
            raise OpenRouterError("OpenRouter n'a retourne aucun choix de reponse.")

        summary = choices[0].get("message", {}).get("content", "").strip()

        if not summary:
            raise OpenRouterError("OpenRouter a retourne une reponse vide.")

        return {
            "summary": summary,
            "model": data.get("model", self.model),
            "usage": data.get("usage"),
        }
