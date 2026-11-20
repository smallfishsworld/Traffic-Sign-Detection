"""Command-line interface: ``tsd <command> --help`` documents every option."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from tsd.net import DEFAULT_BYTES_PER_SECOND, MAX_CONNECTIONS, DownloadPolicy

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

ConnectionsOpt = Annotated[
    int,
    typer.Option(min=1, max=MAX_CONNECTIONS, help="Parallel connections (polite default: 1)."),
]
RateOpt = Annotated[
    float,
    typer.Option(min=0.01, help="Bandwidth cap in MB/s (decimal megabytes) across connections."),
]


def _policy(connections: int, max_mb_per_s: float) -> DownloadPolicy:
    return DownloadPolicy(max_connections=connections, max_bytes_per_second=max_mb_per_s * 1e6)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _open_unit_interval(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise typer.BadParameter("must be strictly between 0 and 1")
    return value


def _optional_open_unit_interval(value: float | None) -> float | None:
    return None if value is None else _open_unit_interval(value)


@app.command("tt100k")
def tt100k_cmd(
    root: Annotated[
        Path, typer.Argument(help="TT100K root (annotations_all.json, train/, test/).")
    ],
    out: Annotated[Path, typer.Argument(help="Output YOLO dataset directory.")],
    classes: Annotated[
        str | None, typer.Option(help="Comma-separated class list (overrides --min-instances).")
    ] = None,
    min_instances: Annotated[int, typer.Option(help="Keep classes with >= N instances.")] = 100,
    count_split: Annotated[
        str, typer.Option(help="Where instances are counted: train or train+test.")
    ] = "train+test",
    val_fraction: Annotated[
        float,
        typer.Option(
            callback=_open_unit_interval, help="Share of TT100K train used as val (0 < f < 1)."
        ),
    ] = 0.1,
    min_box_size: Annotated[float, typer.Option(help="Drop boxes smaller than this (px).")] = 2.0,
    link_mode: Annotated[str, typer.Option(help="symlink, hardlink, copy or none.")] = "symlink",
    workers: Annotated[int, typer.Option(min=1, help="Processes for image-size reads.")] = 2,
    overwrite: Annotated[
        bool, typer.Option(help="Replace a dataset that already exists in OUT.")
    ] = False,
) -> None:
    """Convert TT100K annotations to a YOLO dataset (train / val / test)."""
    from tsd.data import tt100k

    if count_split not in ("train", "train+test"):
        raise typer.BadParameter("count-split must be 'train' or 'train+test'")
    if link_mode not in ("symlink", "hardlink", "copy", "none"):
        raise typer.BadParameter("link-mode must be symlink, hardlink, copy or none")
    config = tt100k.ConvertConfig(
        root=root,
        out=out,
        classes=tuple(c.strip() for c in classes.split(",")) if classes else None,
        min_instances=min_instances,
        count_split=count_split,  # type: ignore[arg-type]
        val_fraction=val_fraction,
        min_box_size=min_box_size,
        link_mode=link_mode,  # type: ignore[arg-type]
        workers=workers,
        overwrite=overwrite,
    )
    try:
        report = tt100k.convert(config)
    except FileExistsError as err:
        raise typer.BadParameter(str(err)) from err
    typer.echo(json.dumps({"classes": len(report.classes), "images": report.images}, indent=2))


@app.command("openimages")
def openimages_cmd(
    out: Annotated[Path, typer.Option(help="Output dataset directory.")] = Path(
        "data/openimages-traffic"
    ),
    metadata_dir: Annotated[
        Path, typer.Option(help="Where missing metadata CSVs are downloaded.")
    ] = Path("data/openimages/metadata"),
    reuse_dir: Annotated[
        list[Path] | None,
        typer.Option(help="Directories with already-downloaded metadata (verified by MD5)."),
    ] = None,
    connections: ConnectionsOpt = 1,
    max_mb_per_s: RateOpt = DEFAULT_BYTES_PER_SECOND / 1e6,
    max_images: Annotated[int | None, typer.Option(help="Only the first N images.")] = None,
    manifest_copy_dir: Annotated[
        Path | None,
        typer.Option(help="Also copy manifest.csv, annotations.json, summary.json here."),
    ] = Path("data/manifests/openimages-traffic"),
    include_verified_negatives: Annotated[
        bool,
        typer.Option(
            help="Also keep images without target boxes that a human verified as negative for a"
            " target class (a second, stricter evaluation-set variant; use another --out)."
        ),
    ] = False,
) -> None:
    """Build the Open Images traffic-sign/stop-sign/traffic-light evaluation set."""
    from tsd.data import openimages

    config = openimages.BuildConfig(
        out=out,
        metadata_dir=metadata_dir,
        reuse_dirs=tuple(reuse_dir or ()),
        policy=_policy(connections, max_mb_per_s),
        max_images=max_images,
        manifest_copy_dir=manifest_copy_dir,
        include_verified_negatives=include_verified_negatives,
    )
    typer.echo(json.dumps(openimages.build(config), indent=2))


@app.command("fetch-assets")
def fetch_assets_cmd(
    names: Annotated[
        list[str] | None, typer.Argument(help="Assets to fetch (default: all pinned assets).")
    ] = None,
    models_dir: Annotated[Path, typer.Option(help="Where .pt weights go.")] = Path("models"),
    data_dir: Annotated[Path, typer.Option(help="Where archives go and are extracted.")] = Path(
        "data"
    ),
    connections: ConnectionsOpt = 1,
    max_mb_per_s: RateOpt = DEFAULT_BYTES_PER_SECOND / 1e6,
) -> None:
    """Download the pinned YOLO11 weights and the COCO128 calibration images (once)."""
    from tsd.assets import ASSETS, extract_jpegs, fetch_asset
    from tsd.net import Downloader

    with Downloader(_policy(connections, max_mb_per_s)) as downloader:
        for name in names or list(ASSETS):
            if name not in ASSETS:
                raise typer.BadParameter(f"unknown asset {name!r}; known: {sorted(ASSETS)}")
            dest = models_dir if name.endswith(".pt") else data_dir / "downloads"
            path = fetch_asset(name, dest, downloader)
            typer.echo(f"{name}: {path}")
            if name == "coco128.zip":
                jpegs = extract_jpegs(path, data_dir / "calibration" / "coco128")
                typer.echo(f"  {len(jpegs)} calibration images in {jpegs[0].parent}")


ParityImagesOpt = Annotated[
    Path,
    typer.Option(help="Directory of .jpg images for the parity gate (searched recursively)."),
]
NumParityImagesOpt = Annotated[int, typer.Option(min=1, help="Images used for parity.")]
ParityThreadsOpt = Annotated[int, typer.Option(min=1, help="Threads for the parity runs.")]
DEFAULT_PARITY_IMAGES = Path("data/openimages-traffic/images/validation")


def _parity_paths(parity_images: Path, num: int) -> list[Path]:
    paths = sorted(parity_images.rglob("*.jpg"))[:num]
    if not paths:
        raise typer.BadParameter(
            f"no parity images in {parity_images}; build the evaluation set (`tsd openimages`),"
            " pass --parity-images DIR, or --skip-parity"
        )
    return paths


def _file_sha256(*models: Path) -> dict[str, str]:
    """SHA-256 of every file of ``models`` (an OpenVINO IR is an .xml plus a .bin)."""
    from tsd.net import file_digest

    files = []
    for m in models:
        files += [m, m.with_suffix(".bin")] if m.suffix == ".xml" else [m]
    return {f.name: file_digest(f, "sha256") for f in files}


def _parity_gate(
    weights: Path, onnx_path: Path, xml: Path, paths: list[Path], imgsz: int, threads: int
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """PyTorch ``weights`` against the ONNX and OpenVINO FP32 exports on ``paths``: the report
    document (tolerances, both comparisons, every compared file's SHA-256, provenance) and the
    failures by runtime (empty: the exports pass)."""
    from tsd.infer.backends import OnnxRuntimeBackend, OpenVINOBackend, TorchBackend
    from tsd.infer.export import PARITY_TOLERANCE, compare_raw_outputs
    from tsd.io import load_image
    from tsd.provenance import provenance

    images = [load_image(p) for p in paths]
    reference = TorchBackend(weights, threads)
    reports = [
        compare_raw_outputs(reference, candidate, images, input_size=(imgsz, imgsz))
        for candidate in (OnnxRuntimeBackend(onnx_path, threads), OpenVINOBackend(xml, threads))
    ]
    failed = {r.candidate: r.failures() for r in reports if r.failures()}
    doc = {
        "weights": weights.name,
        "tolerance": PARITY_TOLERANCE,
        "passed": not failed,
        "images": [p.name for p in paths],
        "files_sha256": _file_sha256(weights, onnx_path, xml),
        "parity": [r.as_dict() for r in reports],
        "provenance": provenance(),
    }
    return doc, failed


def _write_report(doc: dict[str, Any], report: Path | None) -> None:
    text = json.dumps(doc, indent=2)
    typer.echo(text)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text + "\n")


def _install(staged: Path, target: Path) -> None:
    """Move a staged file or directory to ``target``, replacing what is there."""
    import shutil

    if target.is_dir() and not target.is_symlink():
        old = target.with_name(f".{target.name}.replaced")
        shutil.rmtree(old, ignore_errors=True)
        target.replace(old)
        staged.replace(target)
        shutil.rmtree(old)
    else:
        staged.replace(target)


@app.command("export")
def export_cmd(
    weights: Annotated[Path, typer.Argument(help="Ultralytics .pt weights.")],
    out_dir: Annotated[Path, typer.Option(help="Output directory.")] = Path("models"),
    imgsz: Annotated[int, typer.Option(help="Square input size.")] = 640,
    opset: Annotated[int, typer.Option(help="ONNX opset.")] = 18,
    parity_images: ParityImagesOpt = DEFAULT_PARITY_IMAGES,
    skip_parity: Annotated[
        bool, typer.Option(help="Export without the parity gate (not recommended).")
    ] = False,
    num_parity_images: NumParityImagesOpt = 8,
    report: Annotated[Path | None, typer.Option(help="Write the parity report (JSON).")] = None,
    threads: ParityThreadsOpt = 2,
) -> None:
    """Export to ONNX (dynamic batch) and OpenVINO IR (FP32), then gate on parity with PyTorch:
    the exports are written to a staging directory inside OUT_DIR and moved into place only if
    both runtimes' raw outputs and detections are within ``tsd.infer.export.PARITY_TOLERANCE``;
    otherwise nothing is installed and the exit status is 1. ``tsd parity`` re-checks exports
    that are already in place."""
    import shutil
    import tempfile

    from tsd.infer.export import export_onnx, export_openvino

    paths = [] if skip_parity else _parity_paths(parity_images, num_parity_images)
    stem = weights.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{stem}-export-", dir=out_dir))
    try:
        onnx_staged = export_onnx(weights, staging / f"{stem}.onnx", imgsz, opset)
        xml_staged = export_openvino(onnx_staged, staging / f"{stem}_openvino_fp32")
        if skip_parity:
            typer.echo("parity gate skipped")
        else:
            doc, failed = _parity_gate(weights, onnx_staged, xml_staged, paths, imgsz, threads)
            doc["installed"] = not failed
            _write_report(doc, report)
            if failed:
                typer.echo(f"parity gate FAILED, nothing installed: {failed}", err=True)
                raise typer.Exit(1)
        for name in (f"{stem}.onnx", f"{stem}.onnx.json", f"{stem}_openvino_fp32"):
            _install(staging / name, out_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    xml = out_dir / f"{stem}_openvino_fp32" / f"{stem}.xml"
    typer.echo(f"ONNX: {out_dir / f'{stem}.onnx'}\nOpenVINO IR: {xml}")


@app.command("parity")
def parity_cmd(
    weights: Annotated[Path, typer.Argument(help="Ultralytics .pt weights (the reference).")],
    onnx: Annotated[
        Path | None, typer.Option(help="ONNX export (default: <weights dir>/<stem>.onnx).")
    ] = None,
    ir: Annotated[
        Path | None,
        typer.Option(
            help="OpenVINO FP32 IR (default: <weights dir>/<stem>_openvino_fp32/<stem>.xml)."
        ),
    ] = None,
    imgsz: Annotated[int, typer.Option(help="Square input size.")] = 640,
    parity_images: ParityImagesOpt = DEFAULT_PARITY_IMAGES,
    num_parity_images: NumParityImagesOpt = 8,
    report: Annotated[Path | None, typer.Option(help="Write the parity report (JSON).")] = None,
    threads: ParityThreadsOpt = 2,
) -> None:
    """Check-only: run the export parity gate on exports that are already in place (the files a
    benchmark or the service uses), without exporting. The report records every compared
    file's SHA-256; the exit status is 1 if either export is outside the tolerance."""
    onnx_path = onnx or weights.with_suffix(".onnx")
    xml = ir or weights.parent / f"{weights.stem}_openvino_fp32" / f"{weights.stem}.xml"
    for path in (weights, onnx_path, xml, xml.with_suffix(".bin")):
        if not path.is_file():
            raise typer.BadParameter(f"{path} does not exist")
    paths = _parity_paths(parity_images, num_parity_images)
    doc, failed = _parity_gate(weights, onnx_path, xml, paths, imgsz, threads)
    doc["checked"] = {"onnx": str(onnx_path), "openvino": str(xml)}
    _write_report(doc, report)
    if failed:
        typer.echo(f"parity gate FAILED: {failed}", err=True)
        raise typer.Exit(1)


@app.command("quantize")
def quantize_cmd(
    model: Annotated[Path, typer.Argument(help="FP32 OpenVINO IR (.xml) from `tsd export`.")],
    calibration_dir: Annotated[Path, typer.Option(help="Calibration .jpg images.")] = Path(
        "data/calibration/coco128"
    ),
    out_dir: Annotated[
        Path | None, typer.Option(help="Output directory (default: <stem>_openvino_int8).")
    ] = None,
    quantize_head: Annotated[
        bool, typer.Option(help="Also quantize the detect head's box decoding (ablation).")
    ] = False,
    eval_images: Annotated[
        Path,
        typer.Option(
            help="Refuse to calibrate if any calibration image near-duplicates these (must"
            " exist: build the evaluation set first, or pass --skip-eval-check)."
        ),
    ] = Path("data/openimages-traffic/images"),
    skip_eval_check: Annotated[
        bool, typer.Option(help="Skip the near-duplicate check (recorded as skipped).")
    ] = False,
    report: Annotated[Path | None, typer.Option(help="Write a JSON summary here.")] = None,
) -> None:
    """Post-training INT8 quantization (NNCF) of an OpenVINO IR."""
    from dataclasses import asdict

    from tsd.data.dedup import cross_overlap
    from tsd.infer.quantize import quantize_int8
    from tsd.provenance import provenance, warn_if_dirty

    warn_if_dirty()
    images = sorted(calibration_dir.glob("*.jpg"))
    if not images:
        raise typer.BadParameter(f"no .jpg calibration images in {calibration_dir}")
    summary: dict[str, object] = {"model": str(model), "calibration_dir": str(calibration_dir)}
    if skip_eval_check:
        summary["calibration_eval_overlap"] = "skipped (--skip-eval-check)"
    else:
        eval_jpgs = sorted(eval_images.rglob("*.jpg")) if eval_images.is_dir() else []
        if not eval_jpgs:
            raise typer.BadParameter(
                f"no evaluation images in {eval_images} for the calibration near-duplicate"
                " check; build them with `tsd openimages` or pass --skip-eval-check"
            )
        overlap = cross_overlap(images, eval_jpgs)
        summary["calibration_eval_overlap"] = asdict(overlap)
        if overlap.duplicates:
            raise typer.BadParameter(
                f"calibration images near-duplicate the evaluation set: {overlap.duplicates}"
            )
    suffix = "int8_fullq" if quantize_head else "int8"
    stem = model.stem
    target = out_dir or model.parent.parent / f"{stem}_openvino_{suffix}"
    summary.update(quantize_int8(model, images, target / model.name, ignore_head=not quantize_head))
    summary["output"] = str(target / model.name)
    summary["provenance"] = provenance()
    text = json.dumps(summary, indent=2)
    typer.echo(text)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(text + "\n")


def _parse_model_arg(arg: str) -> tuple[Path, str, int]:
    """``path[@bf16][@tiles=SIZE]``: ``@bf16`` runs an OpenVINO FP32 IR with the BF16
    inference-precision hint, ``@tiles=SIZE`` adds sliced inference with SIZE px tiles."""
    path, *suffixes = arg.split("@")
    hint, tiles = "f32", 0
    for suffix in suffixes:
        if suffix == "bf16":
            hint = "bf16"
        elif suffix.startswith("tiles=") and suffix[6:].isdigit():
            tiles = int(suffix[6:])
        else:
            raise typer.BadParameter(f"unknown model suffix @{suffix} (use @bf16, @tiles=SIZE)")
    return Path(path), hint, tiles


@app.command("eval")
def eval_cmd(
    models: Annotated[
        list[str],
        typer.Argument(
            help="Models: .pt, .onnx or OpenVINO .xml; suffixes @bf16 (.xml) and @tiles=SIZE."
        ),
    ],
    data_dir: Annotated[Path, typer.Option(help="Evaluation set from `tsd openimages`.")] = Path(
        "data/openimages-traffic"
    ),
    out: Annotated[Path, typer.Option(help="Report (JSON); a .md table is written next to it.")] = (
        Path("bench/results/eval/eval.json")
    ),
    detections_dir: Annotated[
        Path, typer.Option(help="Where detections are saved (gzipped JSON, committed).")
    ] = Path("bench/results/eval/detections"),
    reuse: Annotated[
        bool,
        typer.Option(
            help="Reuse saved detections whose model files, NMS, tiling and images are unchanged."
        ),
    ] = True,
    replicates: Annotated[int, typer.Option(help="Bootstrap replicates.")] = 1000,
    seed: Annotated[int, typer.Option(help="Bootstrap seed.")] = 0,
    threads: Annotated[int, typer.Option(help="Inference threads.")] = 2,
    tile_overlap: Annotated[float, typer.Option(help="Overlap of @tiles runs.")] = 0.2,
    tile_border_filter: Annotated[
        bool,
        typer.Option(help="Drop truncated tile detections at inner tile borders (@tiles runs)."),
    ] = True,
    tile_merge_iou: Annotated[
        float | None,
        typer.Option(
            help="IoU of the NMS that merges tile and full-image detections (@tiles runs);"
            " default: the detector's own NMS IoU (0.7 in evaluation).",
            callback=_optional_open_unit_interval,
        ),
    ] = None,
    count_unverified: Annotated[
        bool,
        typer.Option(
            help="Strict variant: also count detections of classes nobody verified in an image."
        ),
    ] = False,
) -> None:
    """mAP50 / mAP50-95 with 95% bootstrap intervals on the Open Images evaluation set."""
    from dataclasses import asdict

    from tsd.eval.protocol import EvalSet, map_classes
    from tsd.eval.runner import (
        PredictFn,
        RunDetections,
        can_reuse,
        config_key,
        detector_fn,
        eval_nms,
        markdown_table,
        model_files,
        report,
        run_model,
    )
    from tsd.infer import EVAL_NMS, Detector, NMSConfig, backend_for
    from tsd.infer.tiling import TiledDetector, TilingConfig
    from tsd.net import file_digest
    from tsd.provenance import provenance, warn_if_dirty

    evalset = EvalSet.load(data_dir)
    image_ids = [img.image_id for img in evalset.images]
    runs: list[RunDetections] = []
    files: dict[str, dict[str, str]] = {}
    warn_if_dirty()
    for arg in models:
        path, hint, tiles = _parse_model_arg(arg)
        tiling = (
            TilingConfig(
                tile_size=tiles,
                overlap=tile_overlap,
                border_margin=TilingConfig.border_margin if tile_border_filter else None,
                merge_iou=tile_merge_iou,
            )
            if tiles
            else None
        )
        tag = f"{path.parent.name}_{path.stem}" if backend_for(path) == "openvino" else path.name
        tag += "_bf16" if hint == "bf16" else ""
        tag += f"_tiles{tiles}" if tiles else ""
        extra = {"tiling": asdict(tiling)} if tiling else {}
        cache = detections_dir / f"{tag}.{config_key(EVAL_NMS, extra.get('tiling'), hint)}.json.gz"
        saved = RunDetections.load(cache) if reuse and cache.exists() else None
        if saved is not None:
            current = {p.name: file_digest(p, "sha256") for p in model_files(path)}
            nms = asdict(eval_nms(map_classes(evalset.classes, saved.model_names)))
            if can_reuse(saved, current, extra.get("tiling"), nms, image_ids):
                typer.echo(f"reusing {cache}")
            else:
                typer.echo(f"not reusing {cache}: model files, settings or images differ")
                saved = None
        if saved is None:
            typer.echo(f"predicting with {arg} ...")

            def wrap(det: Detector, nms: NMSConfig, cfg: TilingConfig | None = tiling) -> PredictFn:
                if cfg is None:
                    return detector_fn(det, nms)
                tiled = TiledDetector(det, cfg)
                return lambda image: tiled.predict_one(image, nms)

            run = run_model(
                path, evalset, label="", threads=threads, openvino_precision=hint, wrap=wrap,
                extra={**extra, "provenance": provenance()},
            )  # fmt: skip
            run.label = f"{Path(run.source).stem}/{run.backend}" + (
                f"+tiles{tiles}" if tiles else ""
            )
            run.save(cache)
        loaded = RunDetections.load(cache)  # score exactly what was saved (rounded)
        runs.append(loaded)
        files[loaded.label] = {"path": str(cache), "sha256": file_digest(cache, "sha256")}
    labels = [r.label for r in runs]
    if len(set(labels)) != len(labels):
        raise typer.BadParameter(f"two models produce the same run label: {labels}")
    doc = report(
        evalset,
        runs,
        replicates=replicates,
        seed=seed,
        drop_unverified=not count_unverified,
        detection_files=files,
    )
    doc["provenance"] = provenance()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n")
    table = markdown_table(doc)
    out.with_suffix(".md").write_text(table)
    typer.echo(table)


@app.command("serve")
def serve_cmd(
    model: Annotated[
        Path | None, typer.Option(help="Model file (default: TSD_MODEL_PATH).")
    ] = None,
    host: Annotated[str | None, typer.Option(help="Bind address (default 127.0.0.1).")] = None,
    port: Annotated[int | None, typer.Option(help="Port (default 8000).")] = None,
    max_batch_size: Annotated[
        int | None, typer.Option(help="Micro-batch size; 1 disables batching.")
    ] = None,
    max_wait_ms: Annotated[float | None, typer.Option(help="Micro-batch wait (ms).")] = None,
    threads: Annotated[int | None, typer.Option(help="Inference threads.")] = None,
) -> None:
    """Run the HTTP service (settings from TSD_* environment variables; options override)."""
    import uvicorn

    from tsd.serve import ServeSettings, create_app

    overrides: dict[str, Any] = {
        "model_path": model,
        "host": host,
        "port": port,
        "max_batch_size": max_batch_size,
        "max_wait_ms": max_wait_ms,
        "threads": threads,
    }
    settings = ServeSettings(**{k: v for k, v in overrides.items() if v is not None})
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
