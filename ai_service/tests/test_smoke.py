"""Smoke tests so CI collects at least one test (pytest exits with code 5 if none)."""


def test_python_environment() -> None:
    assert True


def test_parse_symptoms_importable() -> None:
    from ai_service.main import parse_symptoms

    assert parse_symptoms(None) == []
    assert parse_symptoms("  ") == []
    assert parse_symptoms("Fever, cough") == ["fever", "cough"]
