import io
import json
import tarfile
from pathlib import Path

import soundfile as sf


ARCHIVES = {
    "train": "train-clean-100.tar.gz",
    "validation.clean": "dev-clean.tar.gz",
    "validation.other": "dev-other.tar.gz",
    "test.clean": "test-clean.tar.gz",
    "test.other": "test-other.tar.gz",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def read_transcripts(tf: tarfile.TarFile) -> dict[str, str]:
    transcripts = {}
    for member in tf.getmembers():
        if not member.isfile() or not member.name.endswith(".trans.txt"):
            continue
        f = tf.extractfile(member)
        if f is None:
            continue
        for raw_line in f.read().decode("utf-8").splitlines():
            sample_id, text = raw_line.split(" ", 1)
            transcripts[sample_id] = normalize_text(text)
    return transcripts


def build_split(
    archive_path: Path,
    split_name: str,
    output_root: Path,
    export_audio: bool = True,
    max_samples: int | None = None,
) -> None:
    manifest_records = []
    audio_dir = output_root / "audio" / "librispeech" / split_name
    manifest_path = output_root / "manifests" / "librispeech" / f"{split_name}.jsonl"

    ensure_dir(audio_dir)
    ensure_dir(manifest_path.parent)

    print(f"Reading {archive_path.name} -> {split_name}")
    with tarfile.open(archive_path, "r:gz") as tf:
        transcripts = read_transcripts(tf)

        audio_members = [
            member
            for member in tf.getmembers()
            if member.isfile() and member.name.endswith(".flac")
        ]

        for idx, member in enumerate(audio_members):
            if max_samples is not None and len(manifest_records) >= max_samples:
                break

            sample_id = Path(member.name).stem
            text = transcripts.get(sample_id)
            if not text:
                continue

            wav_path = audio_dir / f"{sample_id}.wav"
            audio_path = str(wav_path)
            duration_sec = None
            sampling_rate = 16000

            if export_audio and not wav_path.exists():
                f = tf.extractfile(member)
                if f is None:
                    continue
                audio, sampling_rate = sf.read(io.BytesIO(f.read()), dtype="float32")
                sf.write(wav_path, audio, sampling_rate)
                duration_sec = round(len(audio) / float(sampling_rate), 4)
            elif wav_path.exists():
                info = sf.info(wav_path)
                sampling_rate = int(info.samplerate)
                duration_sec = round(info.frames / float(info.samplerate), 4)

            record = {
                "id": sample_id,
                "dataset": "librispeech",
                "split": split_name,
                "audio_path": audio_path,
                "sampling_rate": sampling_rate,
                "duration_sec": duration_sec,
                "source_lang": "en",
                "target_lang": "uk",
                "source_text": text,
                "target_text_uk": "",
            }
            manifest_records.append(record)

            if (idx + 1) % 1000 == 0:
                print(f"  scanned {idx + 1} audio files, kept {len(manifest_records)}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for record in manifest_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {manifest_path} ({len(manifest_records)} records)")


def main() -> None:
    archive_dir = Path("local_archives")
    output_root = Path("ua_ast_data")

    for split_name, filename in ARCHIVES.items():
        archive_path = archive_dir / filename
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing archive: {archive_path}")
        build_split(
            archive_path=archive_path,
            split_name=split_name,
            output_root=output_root,
            export_audio=True,
        )


if __name__ == "__main__":
    main()
