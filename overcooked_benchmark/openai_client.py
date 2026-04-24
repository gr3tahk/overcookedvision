from __future__ import annotations

import os


_client = None


def get_openai_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def create_chat_completion(client, **kwargs):
    """Call Chat Completions across model families with small parameter differences."""
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as exc:
        message = str(exc)
        if "max_tokens" in message and "max_completion_tokens" in message and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            return client.chat.completions.create(**kwargs)
        if "temperature" in message and "unsupported" in message.lower() and "temperature" in kwargs:
            kwargs.pop("temperature")
            return client.chat.completions.create(**kwargs)
        raise
