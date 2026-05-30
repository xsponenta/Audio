# Audio Coursework

This repository collects all audio coursework tasks in one `main` branch. The
original task branches were imported into separate folders under `tasks/` so
notebooks, reports, metrics, model artifacts, and helper scripts stay grouped by
assignment instead of being mixed in the repository root.

## Structure

| Folder | Task |
| --- | --- |
| `tasks/automatic_speech_recognition/` | ASR and speech translation experiments |
| `tasks/birdclef_2026/` | BirdCLEF 2026 bird audio classification |
| `tasks/generative_audio/` | LLM + TTS generation and xTTS evaluation |
| `tasks/phoneme_recognition/` | Phoneme recognition experiments and PER evaluation |

Each task folder keeps its original notebooks and reports. The root of the
repository is now only an index for the coursework.

## Automatic Speech Recognition

Source branch: `automatic_speech_recognition`

This task contains three related experiments:

- `asr_toronto/` trains and evaluates Ukrainian ASR with Whisper-based models.
- `ast_level1_fleurs/` covers English-to-Ukrainian speech translation work on
  FLEURS.
- `ast_level2_librispeech/` builds a LibriSpeech-based Ukrainian AST pipeline,
  fine-tunes Whisper, and evaluates translation quality with COMET.

Main artifacts:

- `Lab_3_report.pdf` and `report.tex` with the combined report.
- Training and evaluation notebooks for ASR, AST Level 1, and AST Level 2.
- Saved metrics, manifests, tokenizer/config files, and COMET evaluation output.

## BirdCLEF 2026

Source branch: `birdclef_2026`

This task focuses on bird sound classification for the BirdCLEF 2026 setup. The
branch content was moved into `tasks/birdclef_2026/` without changing the
notebooks or report files.

Main artifacts:

- `bird_fn.ipynb` with the BirdCLEF workflow.
- `lab2_raw_audio_pseudo.ipynb` with raw-audio and pseudo-labeling work.
- `testa.ipynb` with additional experiments.
- `lab_report.pdf` with the written report.

## Generative Audio

Source branch: `generative`

This task explores text/audio generation with LLM and TTS tooling, then measures
xTTS output quality with objective and model-based metrics.

Main artifacts:

- `llm_tts_hw.ipynb` with the generation workflow.
- `xtts_metrics.py` for evaluating generated speech.
- `requirements-metrics.txt` with metric dependencies.
- `xtts_metric_results.csv` and `xtts_metric_summary.json` with evaluation
  results.
- `lab4_report.pdf` with the written report.

Recorded summary metrics:

- CER: `0.810553`
- UTMOS: `3.52151`
- SECS: `0.192064`

## Phoneme Recognition

Source branch: `phoneme_recognition`

This task compares phoneme recognition approaches with wav2vec2/data2vec-style
models, several prediction heads, and audio filtering experiments. It keeps both
notebooks and generated evaluation outputs.

Main artifacts:

- `phoneme_recognition_wav2vec.ipynb` and
  `phoneme_recognition_data2vec.ipynb` with model experiments.
- `filters.ipynb` with preprocessing/filtering experiments.
- `per_results_filters.csv` and `results/` with PER evaluation output.
- `weights/` with saved trained heads.
- `vocab.json` and `lab_report.pdf`.

The standard wav2vec2 evaluation summaries process 336 samples per variant and
store predictions for base, TIMIT, bandpass, bandpass + Wiener, and full
preprocessing variants.

## Repository Cleanup

The final `main` branch was organized from the original branches with one clear
commit per task:

- `Add automatic speech recognition task`
- `Add BirdCLEF classification task`
- `Add generative audio task`
- `Add phoneme recognition task`

The top-level README now documents every task, and branch-specific root files
are contained inside their task directories.
