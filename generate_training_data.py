#!/usr/bin/env python3
"""Generate ambisonic training data based on the notebook prototype."""

import argparse
import csv
import json
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.io.wavfile import write as write_wav
from scipy.signal import windows

INT16_MAX = np.iinfo(np.int16).max
MAX_WORKERS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ambisonic training audio and targets.")
    parser.add_argument("--output-root", type=Path, default=Path("generated_training_data"),
                        help="Directory where the dataset will be stored.")
    parser.add_argument("--num-soundscapes", type=int, default=20,
                        help="Number of soundscapes to generate.")
    parser.add_argument("--soundscape-duration", type=float, default=10.0,
                        help="Length of each soundscape in seconds.")
    parser.add_argument("--ambisonics-order", type=int, default=1,
                        help="Ambisonics order (affects channel count).")
    parser.add_argument("--chirp-duration", type=float, default=0.5,
                        help="Duration of the chirp event in seconds.")
    parser.add_argument("--chirp-frequency", type=float, default=3000.0,
                        help="Frequency of the chirp in Hz.")
    parser.add_argument("--chirp-window", type=str, default="blackman",
                        help="Window used to taper the chirp (scipy.signal.windows name).")
    parser.add_argument("--num-events-min", type=int, default=5,
                        help="Minimum number of events per soundscape.")
    parser.add_argument("--num-events-max", type=int, default=10,
                        help="Maximum number of events per soundscape.")
    parser.add_argument("--snr-min", type=float, default=0.0,
                        help="Lower bound for event SNR.")
    parser.add_argument("--snr-max", type=float, default=6.0,
                        help="Upper bound for event SNR.")
    parser.add_argument("--sample-rate", type=int, default=44100,
                        help="Sample rate for generated audio.")
    parser.add_argument("--background-amplitude", type=float, default=1e-4,
                        help="Standard deviation of the background noise generator.")
    parser.add_argument("--dataset-prefix", type=str, default="soundscape",
                        help="Prefix for individual soundscape folders and files.")
    parser.add_argument("--event-spread", type=float, default=0.0,
                        help="Spread parameter passed to AmbiScaper events.")
    parser.add_argument("--ref-db", type=float, default=-40.0,
                        help="Reference dB value assigned to AmbiScaper instances.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for deterministic dataset generation.")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of parallel workers (capped at 8).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Remove existing output directory before generating new data.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_soundscapes <= 0:
        raise ValueError("--num-soundscapes must be positive")
    if args.soundscape_duration <= 0:
        raise ValueError("--soundscape-duration must be positive")
    if args.chirp_duration <= 0 or args.chirp_duration > args.soundscape_duration:
        raise ValueError("--chirp-duration must be positive and not exceed soundscape duration")
    if args.num_events_min <= 0:
        raise ValueError("--num-events-min must be positive")
    if args.num_events_max < args.num_events_min:
        raise ValueError("--num-events-max must be greater or equal to --num-events-min")
    if args.ambisonics_order < 0:
        raise ValueError("--ambisonics-order must be non-negative")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    if args.snr_max < args.snr_min:
        raise ValueError("--snr-max must be greater or equal to --snr-min")
    if not args.dataset_prefix.strip():
        raise ValueError("--dataset-prefix must not be empty")


def format_float_for_name(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return str(value).replace('.', 'p')


def ensure_assets(output_root: Path, sample_rate: int, chirp_frequency: float,
                  chirp_duration: float, chirp_window: str, background_amplitude: float,
                  background_seed: int, num_channels: int, soundscape_duration: float) -> Dict[str, str]:
    assets_dir = output_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    chirp_name = f"chirp_{format_float_for_name(chirp_frequency)}Hz_{format_float_for_name(chirp_duration * 1000)}ms.wav"
    chirp_path = assets_dir / chirp_name
    background_name = f"background_{format_float_for_name(soundscape_duration)}s.wav"
    background_path = assets_dir / background_name

    create_chirp_file(chirp_path, sample_rate, chirp_frequency, chirp_duration, chirp_window)
    create_background_noise(background_path, sample_rate, soundscape_duration, num_channels,
                            background_amplitude, background_seed)

    return {
        "asset_dir": str(assets_dir),
        "chirp_filename": chirp_path.name,
        "background_filename": background_path.name,
    }


def create_chirp_file(path: Path, sample_rate: int, frequency: float,
                      duration: float, window_name: str) -> None:
    num_samples = int(round(sample_rate * duration))
    if num_samples <= 0:
        raise ValueError("Chirp duration too short for given sample rate")

    t = np.linspace(0.0, duration, num_samples, endpoint=False, dtype=np.float64)
    try:
        envelope = windows.get_window(window_name, num_samples, fftbins=True)
    except ValueError as err:
        raise ValueError(f"Invalid chirp window '{window_name}': {err}") from err

    sine = np.sin(2.0 * np.pi * frequency * t)
    signal = sine * envelope
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    audio = np.asarray(signal * INT16_MAX, dtype=np.int16)
    write_wav(path, sample_rate, audio)


def create_background_noise(path: Path, sample_rate: int, duration: float,
                            channels: int, amplitude: float, seed: int) -> None:
    num_samples = int(round(sample_rate * duration))
    if num_samples <= 0:
        raise ValueError("Background duration too short for given sample rate")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, amplitude, size=(num_samples, channels))
    noise = np.clip(noise, -1.0, 1.0)
    audio = np.asarray(noise * INT16_MAX, dtype=np.int16)
    write_wav(path, sample_rate, audio)


def build_worker_config(args: argparse.Namespace, output_root: Path, asset_info: Dict[str, str],
                        num_channels: int) -> Dict[str, Any]:
    prefix = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in args.dataset_prefix.strip())
    return {
        "output_root": str(output_root),
        "asset_dir": asset_info["asset_dir"],
        "chirp_filename": asset_info["chirp_filename"],
        "background_filename": asset_info["background_filename"],
        "soundscape_duration": float(args.soundscape_duration),
        "chirp_duration": float(args.chirp_duration),
        "num_events_min": int(args.num_events_min),
        "num_events_max": int(args.num_events_max),
        "ambisonics_order": int(args.ambisonics_order),
        "sample_rate": int(args.sample_rate),
        "dataset_prefix": prefix,
        "snr_min": float(args.snr_min),
        "snr_max": float(args.snr_max),
        "ref_db": float(args.ref_db),
        "base_seed": int(args.seed),
        "event_spread": float(args.event_spread),
        "num_channels": int(num_channels),
    }


def _generate_soundscape(index: int, config: Dict[str, Any]) -> Dict[str, Any]:
    import ambiscaper
    import jams
    from numpy.random import default_rng

    output_root = Path(config["output_root"])
    asset_dir = Path(config["asset_dir"])
    dataset_prefix = config["dataset_prefix"]
    dest_name = f"{dataset_prefix}_{index:05d}"
    destination_path = output_root / dest_name

    rng = default_rng(config["base_seed"] + index)
    asc = ambiscaper.AmbiScaper(duration=config["soundscape_duration"],
                                 ambisonics_order=config["ambisonics_order"],
                                 fg_path=str(asset_dir),
                                 bg_path=str(asset_dir))
    asc.ref_db = config["ref_db"]

    asc.add_background(source_file=('const', config["background_filename"]),
                       source_time=('const', 0.0))

    num_events = int(rng.integers(config["num_events_min"], config["num_events_max"] + 1))
    max_start = max(config["soundscape_duration"] - config["chirp_duration"], 0.0)
    for _ in range(num_events):
        start_time = float(rng.uniform(0.0, max_start))
        azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        elevation = float(rng.uniform(-np.pi / 2.0, np.pi / 2.0))
        snr = float(rng.uniform(config["snr_min"], config["snr_max"]))

        asc.add_event(source_file=('const', config["chirp_filename"]),
                      source_time=('const', 0.0),
                      event_time=('const', start_time),
                      event_duration=('const', config["chirp_duration"]),
                      event_azimuth=('const', azimuth),
                      event_elevation=('const', elevation),
                      event_spread=('const', config["event_spread"]),
                      snr=('const', snr),
                      pitch_shift=('const', 1.0),
                      time_stretch=('const', 1.0))

    asc.generate(destination_path=str(destination_path),
                 generate_txt=True,
                 disable_instantiation_warnings=True,
                 allow_repeated_source=True)

    source_dir = destination_path / "source"
    if source_dir.exists():
        shutil.rmtree(source_dir)

    jam_path = destination_path / f"{dest_name}.jams"
    jam = jams.load(str(jam_path))
    events: List[Dict[str, Any]] = []
    for annotation in jam.annotations:
        if annotation.namespace != 'ambiscaper_sound_event':
            continue
        for idx, event in enumerate(annotation.data):
            if event.value.get('role') != 'foreground':
                continue
            events.append({
                "index": idx,
                "start": float(event.time),
                "duration": float(event.duration),
                "end": float(event.time + event.duration),
                "azimuth": float(event.value.get('event_azimuth', 0.0)),
                "elevation": float(event.value.get('event_elevation', 0.0)),
                "spread": float(event.value.get('event_spread', config["event_spread"])),
                "snr": float(event.value.get('snr', 0.0)),
            })

    events.sort(key=lambda item: item["start"])
    targets = {
        "soundscape": dest_name,
        "duration": config["soundscape_duration"],
        "sample_rate": config["sample_rate"],
        "events": events,
    }

    target_path = destination_path / f"{dest_name}_targets.json"
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(targets, handle, indent=2)

    audio_path = destination_path / f"{dest_name}.wav"
    txt_path = destination_path / f"{dest_name}.txt"

    return {
        "index": index,
        "audio": str(audio_path.relative_to(output_root)),
        "annotation": str(jam_path.relative_to(output_root)),
        "txt": str(txt_path.relative_to(output_root)),
        "targets": str(target_path.relative_to(output_root)),
        "num_events": len(events),
    }


def write_manifest(output_root: Path, entries: List[Dict[str, Any]]) -> None:
    manifest_path = output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)

    csv_path = output_root / "manifest.csv"
    fieldnames = ["index", "audio", "annotation", "txt", "targets", "num_events"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"Argument error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    output_root = args.output_root.resolve()
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    num_channels = (args.ambisonics_order + 1) ** 2
    asset_info = ensure_assets(output_root, args.sample_rate, args.chirp_frequency,
                               args.chirp_duration, args.chirp_window, args.background_amplitude,
                               args.seed, num_channels, args.soundscape_duration)

    worker_config = build_worker_config(args, output_root, asset_info, num_channels)

    requested_workers = min(args.max_workers, MAX_WORKERS)
    max_workers = max(1, min(requested_workers, args.num_soundscapes))

    results: List[Dict[str, Any]] = []
    errors = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_soundscape, idx, worker_config): idx
                   for idx in range(args.num_soundscapes)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - surfaced to user
                errors += 1
                print(f"Generation failed for index {idx}: {exc}", file=sys.stderr)
            else:
                results.append(result)
                print(f"Generated soundscape {idx:05d} with {result['num_events']} events")

    if errors:
        print(f"Encountered {errors} generation failures", file=sys.stderr)
        raise SystemExit(1)

    results.sort(key=lambda item: item["index"])
    write_manifest(output_root, results)
    print(f"Dataset ready at {output_root}")


if __name__ == "__main__":
    main()
