# Local MLOps Workflow

Python 3.12와 uv가 필요하다. Docker 배치는 Linux 또는 WSL에서 실행한다. 아래 명령은 저장소 루트 기준이다.

```bash
uv sync --locked --group mlops
mkdir -p artifacts
```

`mlops` 그룹은 학습 추적과 오케스트레이션을 실행할 때 설치한다. 기존 CSV CLI와 배치 이미지는 기본 의존성으로 동작한다.

## Track a Model Build

UCI CSV를 준비한 뒤 새 모델 이름으로 학습 실행을 기록한다. 이 명령은 Train에 학습한 모델로 Validation 지표를 계산하고, 같은 설정을 Train+Validation에 새로 fit해 배치 ZIP과 등록용 모델을 만든다. `bank-marketing-v1.zip`은 그대로 보관한다.

```bash
uv run --locked --group mlops python scripts/download_data.py
uv run --locked --group mlops python scripts/track_training.py \
  --data data/raw/bank-additional/bank-additional-full.csv \
  --output models/bank-marketing-candidate-01.zip \
  --model-version candidate-01 \
  --tracking-uri sqlite:///artifacts/mlflow.db \
  --register-as bank-marketing
```

MLflow 실행에는 개발 데이터 지문, 모델 설정, Validation 지표와 Top-K 표, 배치 ZIP이 연결된다. 레지스트리 버전의 메타데이터에서 해당 실행의 ZIP 경로를 확인할 수 있다. 등록한 모델은 후보로 보관하고, 배치에서는 사용할 ZIP을 명시한다. ZIP의 런타임 버전은 실행 환경과 호환되어야 한다.

```bash
uv run --locked --group mlops mlflow ui \
  --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1
```

## Run a Containerized Batch

README Quick Start에서 만든 합성 `artifacts/contacts.csv`와 로컬 `v1` ZIP을 사용한다. Docker 이미지는 Python 3.12.15와 잠금 파일의 런타임 패키지를 설치한다. 입력과 모델은 실행할 때 읽기 전용으로 연결한다.

```bash
docker build --tag bank-marketing-batch:local .

PREFECT_HOME="$PWD/artifacts/prefect" \
PREFECT_SERVER_ANALYTICS_ENABLED=false \
PREFECT_SERVER_EPHEMERAL_ENABLED=true \
uv run --locked --group mlops python scripts/run_batch_flow.py \
  --input artifacts/contacts.csv \
  --model models/bank-marketing-v1.zip \
  --output artifacts/scores-prefect.csv \
  --image bank-marketing-batch:local
```

Prefect는 CLI 실행과 출력 검사를 별도 작업으로 표시한다. 성공 로그에서 입력·출력 행 수, 처리 시간, 결과 크기, `job=unknown` 비율과 점수 중앙값을 확인할 수 있다. CLI가 실패하면 해당 작업이 실패로 표시되고 이후 출력 검사가 실행되지 않는다. 로컬 CLI를 사용하려면 `--image` 옵션을 생략한다. 이전 모델로 재실행할 때는 `--model`과 별도 `--output`을 지정한다.

실행 이력을 대시보드에서 보려면 별도 터미널에서 `PREFECT_HOME="$PWD/artifacts/prefect" uv run --locked --group mlops prefect server start --host 127.0.0.1`을 실행한다. 배치 명령에 `PREFECT_API_URL=http://127.0.0.1:4200/api`를 설정하면 해당 서버에 연결된다.

## Verification Scope

- GitHub Actions 워크플로는 Ruff, pytest, Docker 빌드와 CLI 진입점 검사를 정의한다.
- 배치에서 지표를 확인하는 범위는 로컬 실행이다. 정기 스케줄, 외부 알림과 클라우드 배포는 후속 운영 범위다.
- 실제 라벨 기반 재학습과 교체 기준은 [진행 기록](project-status.md)의 Stage 11에 정리했다.
