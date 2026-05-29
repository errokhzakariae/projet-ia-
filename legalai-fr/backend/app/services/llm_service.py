from fastapi import HTTPException

from src.llm.openrouter_client import OpenRouterClient, OpenRouterError


def summarize_text(text: str, max_tokens: int = 700, temperature: float = 0.2) -> dict:
    try:
        client = OpenRouterClient()
        return client.summarize_legal_text(
            text=text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except OpenRouterError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
