# API Contract

## Core endpoints

- `POST /predict`
- `GET /health`
- `GET /readyz`
- `GET /livez`
- `GET /model/info`
- `GET /models/versions`
- `POST /models/active`

## MLOps endpoints

- `GET /mlops/health`
- `GET /mlops/status`
- `GET /mlops/models`
- `PUT /mlops/models/active`

## Continuous training endpoints

- `POST /continuous-training/feedback`
- `POST /continuous-training/training/trigger`
- `GET /continuous-training/training/status`
