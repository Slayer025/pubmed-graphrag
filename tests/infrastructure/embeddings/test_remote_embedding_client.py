"""Tests for the remote embedding client abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from src.infrastructure.embeddings.remote_embedding_client import (
    RemoteEmbeddingClient,
    create_embedding_client,
)


class _FakeModel:
    """Fake sentence-transformers model returning deterministic vectors."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts: list[str], *, batch_size: int = 64, **_: object) -> np.ndarray:
        rng = np.random.default_rng(0)
        vectors = rng.random((len(texts), self.dim)).astype(np.float32)
        return vectors


def _fake_create_embedding_model(
    model_name: str = "",
    *,
    cache_folder: str | None = None,
) -> _FakeModel:
    del model_name, cache_folder
    return _FakeModel(dim=4)


@pytest.fixture
def patch_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.infrastructure.embeddings.remote_embedding_client.create_embedding_model",
        _fake_create_embedding_model,
    )


def test_local_embed_returns_normalized_vectors(patch_local_model: None) -> None:
    client = RemoteEmbeddingClient(provider="local", model_name="fake")
    vectors = client.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 4
    norms = [np.linalg.norm(v) for v in vectors]
    assert all(abs(n - 1.0) < 1e-6 for n in norms)


def test_embed_query_returns_single_vector(patch_local_model: None) -> None:
    client = RemoteEmbeddingClient(provider="local", model_name="fake")
    vector = client.embed_query("query")
    assert isinstance(vector, list)
    assert len(vector) == 4


def test_empty_embed_returns_empty_list(patch_local_model: None) -> None:
    client = RemoteEmbeddingClient(provider="local", model_name="fake")
    assert client.embed([]) == []


def test_remote_http_without_url_falls_back_to_local(patch_local_model: None) -> None:
    client = RemoteEmbeddingClient(provider="remote_http", model_name="fake")
    assert client.provider == "local"
    assert client.fallback_reason is not None
    vectors = client.embed(["hello"])
    assert len(vectors) == 1


def test_huggingface_api_without_token_falls_back_to_local(patch_local_model: None) -> None:
    client = RemoteEmbeddingClient(provider="huggingface_api", model_name="fake")
    assert client.provider == "local"
    assert client.fallback_reason is not None


def test_create_embedding_client_factory_reports_provider(patch_local_model: None) -> None:
    result = create_embedding_client(provider="local", model_name="fake")
    assert result.provider == "local"
    assert result.selected_provider == "local"
    assert result.fallback_reason is None


def test_remote_http_embed_uses_service_url(
    patch_local_model: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {}

    class _FakeClient:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def post(self, url: str, *, json: dict, **_: object) -> "_FakeResponse":
            called["url"] = url
            called["json"] = json
            return _FakeResponse()

    class _FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"embeddings": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]}

    monkeypatch.setattr(
        "src.infrastructure.embeddings.remote_embedding_client.httpx.Client",
        _FakeClient,
    )

    client = RemoteEmbeddingClient(
        provider="remote_http",
        model_name="fake",
        service_url="http://localhost:8000/embed",
    )
    vectors = client.embed(["a", "b"])
    assert called["url"] == "http://localhost:8000/embed"
    assert called["json"]["texts"] == ["a", "b"]
    assert len(vectors) == 2
    assert vectors[0] == [1.0, 0.0, 0.0, 0.0]


def test_remote_failure_falls_back_to_local(
    patch_local_model: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingClient:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "_FailingClient":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def post(self, **_: object) -> "_FailingClient":
            raise RuntimeError("network down")

    monkeypatch.setattr(
        "src.infrastructure.embeddings.remote_embedding_client.httpx.Client",
        _FailingClient,
    )

    client = RemoteEmbeddingClient(
        provider="remote_http",
        model_name="fake",
        service_url="http://localhost:8000/embed",
    )
    vectors = client.embed(["hello"])
    assert client.provider == "local"
    assert client.fallback_reason is not None
    assert len(vectors) == 1
