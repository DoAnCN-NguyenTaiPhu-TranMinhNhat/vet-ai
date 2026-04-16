"""Basic smoke tests for app routers."""


def test_python_environment() -> None:
    assert True


def test_parse_symptoms_importable() -> None:
    from ai_service.app.api.routers.predict import parse_symptoms

    assert parse_symptoms(None) == []
    assert parse_symptoms("  ") == []
    assert parse_symptoms("Fever, cough") == ["fever", "cough"]


def test_health_router_endpoints(client) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert client.get("/livez").status_code == 200


def test_predict_router_endpoint_registered(client) -> None:
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert "model_version" in body


def test_mlops_router_endpoint_registered(client) -> None:
    response = client.get("/mlops/health")
    assert response.status_code == 200
