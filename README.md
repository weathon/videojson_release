# VideoJSON

Code for **VideoJSON: A Minimal Harness Beats Purpose-Built Memory for Ultra-Long Video Understanding** ([paper](https://cspaper.org/openprint/20261120.0001v1.pdf)).

A vision-language model turns each minute of video into one JSON record. At query time a coding agent writes Python or Bash programs in a sandboxed Docker container to search these records, and answers the question without seeing the video.

Artifacts:

- Indices, MediaStorm-Livestream and the fine-tuning data: https://huggingface.co/datasets/weathon/videojson_release
- LoRA adapter of the fine-tuned Qwen-3.5-Flash: https://huggingface.co/weathon/videojson_release_lora

## Layout

| Path | Content |
|---|---|
| `src/main.py` | The harness: loads a benchmark, starts one container per question, runs the agent, scores the answer |
| `prompts/` | Every prompt the harness uses: caption schema, dataset instructions, tool descriptions, judge prompt, caption prompt |
| `docker/sandbox.Dockerfile` | Image of the question containers |
| `docker/laya.Dockerfile`, `prompts/decision/` | The decision model service and its in-container library (`+ Decision API`) |
| `runs/ablation/*_keys.txt` | The fixed 200-question subsets of the ablation and scaling experiments |
| `scripts/run_v5.sh` | The main-table setting on the five datasets |
| `scripts/caption_old/` | Indexing: ASR, one-minute captions, minute index, hourly index |
| `scripts/index_time_ours.sh`, `scripts/plot_index_time.py` | Indexing wall time |
| `scripts/cost.py` | Score and cost per question of a run |
| `scripts/code_analysis.py`, `scripts/plot_code_nature_ring.py`, `scripts/plot_token_pie.py` | Code and trace analysis |
| `distill/` | Fine-tuning on VideoJSON trajectories |

Paths of the benchmarks, videos and model servers are constants in the files (`/raid/...`, `127.0.0.1:<port>`); change them to yours. `src/main.py` and most scripts end with `1 / 0` after their last line, so a finished run prints `DONE` and then exits with `ZeroDivisionError`.

## Setup

```bash
pip install -r requirements.txt
docker build -f docker/sandbox.Dockerfile -t videojson-sandbox:1 docker
hf download weathon/videojson_release --repo-type dataset --local-dir .
```

The download puts the indices in `data/` (captions by Qwen-3.8-Flash-Next) and `data_cap35/` (captions by Qwen-3.5-Flash), MediaStorm-Livestream in `mediastorm_livestream/` and the fine-tuning data in `distill_data/`.

The benchmarks are not included. `src/main.py` reads them from the paths at its top: MM-Lifelong, EgoLifeQA (`EgoLifeQA_A1_JAKE.json`), Ego-R1-Bench and EgoMemReason. `data/egolife/timelines/` holds the video file paths that `video_qa` reads.

The policy model is served with an OpenAI-compatible API (`--model`, `--base-url`). MM-Lifelong answers are judged by `openai/gpt-5` through OpenRouter with the key in `OPENROUTER_API_KEY_PERSONAL` of `.env`. Runs log to wandb.

## Running the harness

```bash
python src/main.py --dataset egolifeqa --run-name main --concurrency 75 \
    --model deepseek-v4.1-flash-1017 --base-url http://127.0.0.1:8400/v1 --reasoning-effort xhigh \
    --no-video-qa --no-hours
```

`--dataset` is one of `egolifeqa`, `egor1`, `mm_week`, `mm_day`, `mm_month` (`--split val`), `memreason`. Each question writes `runs/<dataset>/<run-name>/q<id>/` with `answer.json` (answer, status, tokens of every model call), `messages.json` (the trace) and `judge.json`; a rerun skips finished questions. `memreason` writes `submission.json`.

| Setting | Flags |
|---|---|
| Minimal Harness (main results) | `--no-video-qa --no-hours` |
| + video_qa | `--no-hours` |
| + Decision API | `--no-video-qa --no-hours --skill prompts/decision/decision_search.md --decision-dir prompts/decision/lib --container-network <network>` |
| + Hourly Caption | `--no-video-qa` |
| w/ MAGIC Index | `--no-video-qa --no-hours --magic-index <dir>` |
| Dense Caption Only | `--no-video-qa --no-hours --dense-caption-only` |
| BM25 Only | `--no-video-qa --no-hours --bm25 --bm25-calls` |
| Without exec_qa | `--no-video-qa --no-hours --no-exec-qa` |
| Bilingual questions and options (every Ego-R1-Bench run; EgoLifeQA in the bilingual comparison) | `--bilingual` |
| 200-question subset | `--keys runs/ablation/<dataset>_keys.txt` |
| Captions by Qwen-3.5-Flash | `--data-dir data_cap35` |

The Qwen policy models are run with `--temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty <value of the model card> --max-tokens 16384` and without `--reasoning-effort`.

`python scripts/cost.py runs/<dataset>/<run-name> [keys file]` prints the score and the cost per question of a run at US$0.15 per million uncached input tokens, US$0.003 per million cached input tokens and US$0.60 per million output tokens.

The forwarding service between the question containers and the decision model (`+ Decision API`) is not in this repository.

## Indexing

`scripts/caption_old/` holds the indexing code. It reads the videos through media maps and paths of our machines (the constants at the top of `subset_clips.py`, `harness.py` and `egolife_people.py`).

1. `asr_subsets.py`: Qwen3-ASR-1.7B transcript of every one-minute clip.
2. `caption_subset.py <day|month|week> <concurrency> --model <captioner> --ports <ports>`: one JSON caption per clip from 60 frames at 1 fps and the transcript, with the schema `prompts/egolife/vsi_caption_schema.json` and the prompt `prompts/caption/caption_prompt.txt`. `week` adds the transcripts released with EgoLife. `caption_egolife.py` captions the other EgoLife wearers. The default captioner is Qwen-3.8-Flash-Next; `data_cap35/` and the indexing wall time use `--model Qwen/Qwen3.5-35B-A3B-FP8`.
3. `build_data.py`: the minute index (`data/egolife/<wearer>.json`, `data/mm_lifelong/<subset>.json`).
4. `hoursly_index.py`, `hoursly_index_mm_lifelong.py`: the hourly index (`*_hourly.json`), used by `+ Hourly Caption` and by `distill/harness/`.

`scripts/index_time_ours.sh` times steps 1 to 3 on the first minutes of MM-Lifelong Day.

## Fine-tuning on VideoJSON trajectories

| Path | Content |
|---|---|
| `distill/harness/src/main_qwenjudge.py` | The harness revision that collected the teacher trajectories on MediaStorm-Livestream (`--external-month-dir mediastorm_livestream`) |
| `distill/harness/src/main_qwenjudge_recovery.py` | The harness revision that evaluated the base and fine-tuned models on MM-Lifelong |
| `distill/harness/prompts/` | The prompts of these two files |
| `distill/prepare_data.py` | Splits the questions by time and exports the training examples (`distill_data/`) |
| `distill/train_lora.py` | LoRA fine-tuning of Qwen-3.5-Flash |
| `distill/judge/runner.py`, `distill/judge/paired_analysis.py` | GPT-5 scoring of both models' answers and the paired bootstrap |

`distill/harness/` is an earlier revision of the harness with its own prompts; it reads `data/` and `prompts/` next to its `src/`.

## License

WTFPL, see `LICENSE`.
