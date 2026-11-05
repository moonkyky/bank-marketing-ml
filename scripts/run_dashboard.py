"""Open the local Gradio dashboard for the bank marketing batch model."""

import os
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import gradio as gr
import pandas as pd
from simulate_monitoring import make_reference

from bank_marketing_ml.cli import _write_results
from bank_marketing_ml.inference import score_csv

CSS = """
.gradio-container { max-width: 1220px !important; margin: auto !important; }
#hero { background: linear-gradient(116deg, #10233b, #123f46);
  color: #f4f8fa; border-radius: 22px; padding: 32px 38px; margin: 16px 0 22px; }
#hero .eyebrow { color: #93e4ce; font-size: 13px; letter-spacing: .15em; }
#hero h1 { color: #fff; font-size: clamp(32px, 5vw, 54px); margin: 7px 0 6px; }
#hero p { color: #d5e6ed; margin: 0; }
.panel { border-radius: 18px !important; }
.kpis { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.kpi { border: 1px solid var(--border-color-primary); padding: 12px 14px;
  background: var(--background-fill-secondary); border-radius: 12px; }
.kpi span { display: block; font-size: 13px; color: var(--body-text-color-subdued); }
.kpi strong { display: block; margin-top: 4px; font-size: 24px; }
@media (max-width: 520px) { #hero { padding: 24px; } .kpi { padding: 8px; }
  .kpi strong { font-size: 18px; } }
"""

HEADERS = ["순위", "입력 행", "점수"]


def summary(rows: str = "—", best: str = "—", average: str = "—") -> str:
    cards = (("처리 레코드", rows), ("최고 점수", best), ("평균 점수", average))
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><span>{label}</span><strong>{value}</strong></div>'
            for label, value in cards
        )
        + "</div>"
    )


def build_dashboard(workspace: Path, model_dir: Path) -> gr.Blocks:
    models = {
        path.name: path
        for path in sorted(model_dir.glob("*.zip"))
        if path.is_file() and not path.is_symlink()
    }
    example, _ = make_reference()
    example_path = workspace / "example.csv"
    example.head(12).to_csv(example_path, sep=";", index=False)
    default_model = next(iter(models), None)
    if "bank-marketing-v1.zip" in models:
        default_model = "bank-marketing-v1.zip"

    def score(source_kind, uploaded_file, model_name, separator):
        if model_name not in models:
            return summary(), [], None, "모델을 준비한 뒤 화면을 다시 실행하세요."
        if source_kind == "합성 예제":
            source, separator = example_path, ";"
        elif uploaded_file:
            source = uploaded_file
        else:
            return summary(), [], None, "처리할 CSV를 업로드하세요."

        try:
            scores = score_csv(source, models[model_name], sep=separator).to_numpy()
            result_dir = workspace / uuid4().hex
            result_dir.mkdir()
            result_path = result_dir / "scores.csv"
            _write_results(result_path, scores)
        except (OSError, ValueError) as exc:
            return summary(), [], None, f"순위 계산 실패 · {exc}"

        ranked = pd.read_csv(result_path, float_precision="round_trip").nsmallest(
            10, "rank"
        )
        top_rows = [
            [int(rank), int(row), round(float(value), 6)]
            for rank, row, value in ranked[["rank", "input_row", "score"]].itertuples(
                index=False, name=None
            )
        ]
        return (
            summary(f"{len(scores):,}", f"{scores.max():.4f}", f"{scores.mean():.4f}"),
            top_rows,
            result_path,
            f"완료 · {len(scores):,}건 · {Path(source).name} · {model_name}",
        )

    with gr.Blocks(title="Bank Marketing ML", analytics_enabled=False) as dashboard:
        gr.HTML(
            '<header id="hero"><span class="eyebrow">LOCAL BATCH WORKSPACE</span>'
            "<h1>Bank Marketing ML</h1>"
            "<p>정기예금 캠페인 명단을 불러와 연락 우선순위를 확인하세요.</p></header>"
        )
        with gr.Row():
            with gr.Column(
                scale=2, min_width=290, variant="panel", elem_classes="panel"
            ):
                gr.Markdown("### 01 · 배치 설정")
                source_kind = gr.Radio(
                    ["합성 예제", "CSV 업로드"], value="합성 예제", label="입력 명단"
                )
                with gr.Group(visible=False) as upload_options:
                    upload = gr.File(
                        label="명단 CSV", file_types=[".csv"], type="filepath"
                    )
                    separator = gr.Radio(
                        [("세미콜론 (;)", ";"), ("쉼표 (,)", ",")],
                        value=";",
                        label="입력 구분자",
                    )
                model = gr.Dropdown(
                    list(models), value=default_model, label="로컬 모델"
                )
                if not models:
                    gr.Markdown(
                        "`models/`에 호환 모델 ZIP을 준비한 뒤 앱을 다시 실행하세요."
                    )
                run = gr.Button(
                    "순위 계산", variant="primary", interactive=bool(models)
                )
                status = gr.Textbox(label="실행 상태", interactive=False)
            with gr.Column(
                scale=3, min_width=320, variant="panel", elem_classes="panel"
            ):
                gr.Markdown("### 02 · 배치 결과")
                kpis = gr.HTML(summary())
                table = gr.Dataframe(
                    headers=HEADERS,
                    datatype=["number", "number", "number"],
                    label="연락 우선순위 · 상위 10건",
                    interactive=False,
                )
                download = gr.DownloadButton(
                    "전체 결과 CSV 다운로드", interactive=False
                )
                gr.Markdown("다운로드 파일은 입력 행 순서를 유지합니다.")
        source_kind.change(
            lambda kind: gr.update(visible=kind == "CSV 업로드"),
            inputs=source_kind,
            outputs=upload_options,
        )

        outputs = [kpis, table, download, status]

        def reset_results():
            return summary(), [], gr.update(value=None, interactive=False), ""

        gr.on(
            triggers=[
                source_kind.change,
                upload.change,
                model.change,
                separator.change,
            ],
            fn=reset_results,
            outputs=outputs,
            concurrency_id="batch-results",
            api_visibility="private",
        )

        def on_run(source_kind, upload, model, separator):
            cards, rows, result, message = score(source_kind, upload, model, separator)
            return (
                cards,
                rows,
                gr.update(value=result, interactive=result is not None),
                message,
            )

        run.click(
            on_run,
            inputs=[source_kind, upload, model, separator],
            outputs=outputs,
            concurrency_id="batch-results",
            api_name="score_batch",
        )
    return dashboard


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parents[1] / "models"
    with TemporaryDirectory(prefix="bank-marketing-ui-") as temporary:
        workspace = Path(temporary)
        os.environ["GRADIO_TEMP_DIR"] = str(workspace / "cache")
        dashboard = build_dashboard(workspace, model_dir)
        dashboard.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=False,
            max_file_size="5mb",
            blocked_paths=[model_dir],
            css=CSS,
            theme=gr.themes.Soft(
                primary_hue="teal",
                neutral_hue="slate",
                font=("system-ui", "sans-serif"),
                font_mono=("monospace",),
            ),
            footer_links=[],
        )


if __name__ == "__main__":
    main()
