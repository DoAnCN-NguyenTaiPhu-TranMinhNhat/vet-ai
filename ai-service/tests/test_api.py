def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info_loaded(client) -> None:
    resp = client.get("/model/info")
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["loaded"] is True
    assert payload["model_type"]
    assert isinstance(payload.get("classes"), list)
    assert len(payload["classes"]) >= 2


def test_predict_valid_payload(client) -> None:
    payload = {
        "animal_type": "Dog",
        "gender": "Male",
        "age_months": 24,
        "weight_kg": 10.5,
        "temperature": 39.2,
        "heart_rate": 120,
        "current_season": "Summer",
        "vaccination_status": "Fully Vaccinated",
        "medical_history": "Unknown",
        "symptoms_list": "coughing, fever",
        "symptom_duration": 3,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    out = resp.json()
    assert isinstance(out.get("diagnosis"), str)
    assert "confidence" in out
    assert "top_k" in out


def test_predict_missing_required_field_returns_422(client) -> None:
    payload = {
        "animal_type": "Dog",
        "gender": "Male",
        "age_months": 24,
        "weight_kg": 10.5,
        "temperature": 39.2,
        "heart_rate": 120,
        "current_season": "Summer",
        # missing vaccination_status
        "medical_history": "Unknown",
        "symptoms_list": "coughing, fever",
        "symptom_duration": 3,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_empty_symptoms_ok(client) -> None:
    payload = {
        "animal_type": "Cat",
        "gender": "Female",
        "age_months": 36,
        "weight_kg": 4.2,
        "temperature": 38.5,
        "heart_rate": 140,
        "current_season": "Winter",
        "vaccination_status": "Unvaccinated",
        "medical_history": "Unknown",
        "symptoms_list": "",
        "symptom_duration": 1,
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    out = resp.json()
    assert isinstance(out.get("diagnosis"), str)
