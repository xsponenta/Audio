"""Objective TTS metrics used in the XTTS evaluation setup.

Metrics:
- CER: Whisper ASR transcript vs. target text, with punctuation removed.
- UTMOS: naturalness MOS predicted by tarepan/SpeechMOS UTMOS strong.
- SECS: cosine similarity between ECAPA2 speaker embeddings.
"""

import argparse
import csv
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from jiwer import cer


PUNCT_TABLE = str.maketrans("", "", string.punctuation + ".,!?;:\"'`’‘“”…—–-")


def normalize_for_cer(text: str) -> str:
    """Match the XTTS evaluation convention: remove punctuation before CER."""

    text = text.lower().translate(PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """Load mono audio as a float tensor with shape (time,)."""

    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.contiguous(), sr


@dataclass
class TTSMetricModels:
    """Lazy holder for the heavy metric models."""

    device: str | None = None
    whisper_model_id: str = "openai/whisper-large-v3"
    ecapa2_repo_id: str = "Jenthe/ECAPA2"
    ecapa2_filename: str = "ecapa2.pt"

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._asr = None
        self._utmos = None
        self._ecapa2 = None

    @property
    def asr(self):
        if self._asr is None:
            from transformers import pipeline

            device_id = 0 if self.device.startswith("cuda") else -1
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model_id,
                device=device_id,
            )
        return self._asr

    @property
    def utmos(self):
        if self._utmos is None:
            self._utmos = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0",
                "utmos22_strong",
                trust_repo=True,
            ).to(self.device)
            self._utmos.eval()
        return self._utmos

    @property
    def ecapa2(self):
        if self._ecapa2 is None:
            from huggingface_hub import hf_hub_download

            model_file = hf_hub_download(
                repo_id=self.ecapa2_repo_id,
                filename=self.ecapa2_filename,
            )
            self._ecapa2 = torch.jit.load(model_file, map_location=self.device)
        return self._ecapa2


def compute_cer(
    reference_text: str,
    generated_audio_path: str | Path,
    models: TTSMetricModels,
) -> dict[str, float | str]:
    wav, sr = load_audio(generated_audio_path, target_sr=16_000)
    audio = {"array": wav.cpu().numpy(), "sampling_rate": sr}
    hyp = models.asr(audio)["text"]
    ref_norm = normalize_for_cer(reference_text)
    hyp_norm = normalize_for_cer(hyp)
    return {
        "cer": float(cer(ref_norm, hyp_norm)),
        "reference_normalized": ref_norm,
        "hypothesis": hyp,
        "hypothesis_normalized": hyp_norm,
    }


def compute_utmos(generated_audio_path: str | Path, models: TTSMetricModels) -> float:
    wav, sr = load_audio(generated_audio_path)
    wav = wav.unsqueeze(0).to(models.device)
    with torch.no_grad():
        score = models.utmos(wav, sr)
    return float(score.squeeze().detach().cpu())


def _speaker_embedding(path: str | Path, models: TTSMetricModels) -> torch.Tensor:
    wav, _ = load_audio(path, target_sr=16_000)
    wav = wav.unsqueeze(0).to(models.device)
    with torch.no_grad(), torch.jit.optimized_execution(False):
        emb = models.ecapa2(wav)
    return emb.flatten().float()


def compute_secs(
    reference_audio_path: str | Path,
    generated_audio_path: str | Path,
    models: TTSMetricModels,
) -> float:
    ref_emb = _speaker_embedding(reference_audio_path, models)
    gen_emb = _speaker_embedding(generated_audio_path, models)
    return float(F.cosine_similarity(ref_emb, gen_emb, dim=0).detach().cpu())


def evaluate_sample(sample: dict[str, str], models: TTSMetricModels) -> dict[str, object]:
    """Evaluate one row with text, reference_audio, and generated_audio keys."""

    generated_audio = sample["generated_audio"]
    reference_audio = sample["reference_audio"]
    cer_result = compute_cer(sample["text"], generated_audio, models)
    return {
        **sample,
        "CER": cer_result["cer"],
        "UTMOS": compute_utmos(generated_audio, models),
        "SECS": compute_secs(reference_audio, generated_audio, models),
        "asr_hypothesis": cer_result["hypothesis"],
    }


def evaluate_samples(
    samples: Iterable[dict[str, str]],
    models: TTSMetricModels | None = None,
) -> list[dict[str, object]]:
    models = models or TTSMetricModels()
    return [evaluate_sample(sample, models) for sample in samples]


def summarize_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "CER": float(np.mean([row["CER"] for row in rows])),
        "UTMOS": float(np.mean([row["UTMOS"] for row in rows])),
        "SECS": float(np.mean([row["SECS"] for row in rows])),
    }


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CER, UTMOS, and SECS.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default="xtts_metric_results.csv")
    parser.add_argument("--summary-json", default="xtts_metric_summary.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows = evaluate_samples(read_csv(args.input_csv), TTSMetricModels(device=args.device))
    write_csv(args.output_csv, rows)
    summary = summarize_metrics(rows)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
