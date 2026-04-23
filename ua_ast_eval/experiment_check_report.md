# UA AST Experiment Check Report

Checked on: 2026-05-06

## Dataset

Source dataset: LibriSpeech ASR, `train.clean.100`, `validation.clean`, `validation.other`, `test.clean`, `test.other`.

Created dataset: English audio + English source transcript + machine-generated Ukrainian reference translation.

Split sizes:

| Split | Records |
|---|---:|
| train | 28,539 |
| validation.clean | 2,703 |
| validation.other | 2,864 |
| test.clean | 2,620 |
| test.other | 2,939 |

## Data Leakage Check

Pairwise split intersection was checked by:

- `id`
- `audio_path`
- normalized English `source_text`
- normalized Ukrainian `target_text_uk`
- SHA1 hash of the actual audio file bytes

Result:

- Train vs validation/test audio ID overlap: 0
- Train vs validation/test audio path overlap: 0
- Train vs validation/test audio SHA1 overlap: 0
- Validation vs test audio SHA1 overlap: 0
- `test.clean` vs `test.other` audio SHA1 overlap: 0

There are a few repeated normalized text strings across splits, for example short/common utterances such as "yes". This is not audio leakage because the corresponding utterance ids, audio paths, and audio hashes are different.

Conclusion: no audio-level leakage was found between train, validation, and test splits.

## Baseline and Fine-tuned Evaluation

Metric: COMET `Unbabel/wmt22-comet-da`

Baseline model: `openai/whisper-base`

Fine-tuned model: `models/whisper-base-ua-ast`

Training setup:

- `run_mode`: `final_full`
- train samples: all 28,539
- epochs: 3
- max test samples: all

Results:

| Split | Records | Baseline COMET | Fine-tuned COMET | Delta |
|---|---:|---:|---:|---:|
| test.clean | 2,620 | 0.3209 | 0.4174 | +0.0965 |
| test.other | 2,939 | 0.3266 | 0.4139 | +0.0873 |

Conclusion: the fine-tuned model improves COMET on both official LibriSpeech test splits.

## Existing Ukrainian Version Check

Searches performed on 2026-05-06:

- `Ukrainian translation LibriSpeech dataset Ukrainian LibriSpeech parallel corpus`
- `site:huggingface.co/datasets Ukrainian LibriSpeech translation dataset`
- `site:huggingface.co/datasets "LibriSpeech" "Ukrainian" "translation"`
- `"український переклад" "LibriSpeech"`
- `"LibriSpeech" "українською" датасет`

No open dataset matching "LibriSpeech English audio/transcript -> Ukrainian text translation" was found in these searches. Public results found English LibriSpeech ASR datasets and unrelated Ukrainian ASR models/datasets, but not an existing Ukrainian translation of LibriSpeech transcripts.

Important limitation: search results cannot prove absolute non-existence, but no public Ukrainian LibriSpeech translation was found from the checked public sources.

## Baseline Model Pretraining Leakage Note

OpenAI states Whisper was trained on 680,000 hours of multilingual/multitask supervised web data and was not fine-tuned to a specific dataset. OpenAI also explicitly discusses LibriSpeech as a benchmark for Whisper, but the exact pretraining corpus is not fully public. Therefore, we cannot strictly prove that Whisper never saw any LibriSpeech-derived audio/text during pretraining.

What is clean in this experiment:

- The fine-tuning train split has no audio overlap with validation or test.
- The baseline and fine-tuned model are compared on the same held-out test files.
- The observed COMET gains measure the effect of fine-tuning on the created Ukrainian AST dataset under the same test conditions.

For a stricter future experiment, use an additional test set not commonly used in Whisper-era benchmarking, or manually collect a small private held-out test set.
