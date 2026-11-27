"""Build a `docker load`-compatible image archive without Docker or root.

Base: python:3.12-slim-bookworm (linux/amd64) pulled from Docker Hub's registry API.
One extra layer: /app (contest build context app/), numpy 1.26.4 wheel and contest_harness SDK unpacked into
/usr/local/lib/python3.12/site-packages. Config: same Env/WorkingDir/User/Entrypoint as contest/Dockerfile.

  python3 build_image_nodocker.py <build context dir with app/ and sdk/> <out.tar> [--tag zhixun-rsi:v1]
Needs `requests` (any version). Writes the uncompressed docker-save layout: manifest.json, <config>.json, <id>/layer.tar.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO, TAG = "library/python", "3.12-slim-bookworm"
NUMPY = "1.26.4"
SITE = "usr/local/lib/python3.12/site-packages"
ENV_ADD = ["PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
           "MKL_NUM_THREADS=1", "BASELINE_CPU_THREADS=1"]
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json", "application/vnd.oci.image.manifest.v1+json"])


def log(msg):
    print(f"[image] {msg}", file=sys.stderr, flush=True)


def registry():
    token = requests.get("https://auth.docker.io/token", params={"service": "registry.docker.io",
                         "scope": f"repository:{REPO}:pull"}, timeout=60).json()["token"]
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


def get_json(session, ref):
    r = session.get(f"https://registry-1.docker.io/v2/{REPO}/manifests/{ref}", headers={"Accept": ACCEPT}, timeout=60)
    r.raise_for_status()
    return r.json()


def blob(session, digest):
    r = session.get(f"https://registry-1.docker.io/v2/{REPO}/blobs/{digest}", timeout=600)
    r.raise_for_status()
    data = r.content
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"digest mismatch for {digest}")
    return data


def add_dir(tar, name, seen):
    parts = name.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        d = "/".join(parts[:i])
        if d and d not in seen:
            info = tarfile.TarInfo(d)
            info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o755, 0
            tar.addfile(info)
            seen.add(d)


def add_file(tar, name, data, seen, mode=0o644):
    add_dir(tar, os.path.dirname(name), seen)
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    tar.addfile(info, io.BytesIO(data))


def numpy_wheel():
    meta = requests.get(f"https://pypi.org/pypi/numpy/{NUMPY}/json", timeout=60).json()
    want = f"numpy-{NUMPY}-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    item = next(u for u in meta["urls"] if u["filename"] == want)
    data = requests.get(item["url"], timeout=600).content
    if hashlib.sha256(data).hexdigest() != item["digests"]["sha256"]:
        raise ValueError("numpy wheel digest mismatch")
    return data


def app_layer(context):
    buf = io.BytesIO()
    seen = set()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted((context / "app").rglob("*")):
            if path.is_file():
                add_file(tar, "app/" + path.relative_to(context / "app").as_posix(), path.read_bytes(), seen)
        for path in sorted((context / "sdk" / "contest_harness").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                rel = path.relative_to(context / "sdk").as_posix()
                add_file(tar, f"{SITE}/{rel}", path.read_bytes(), seen)
        add_file(tar, f"{SITE}/contest_harness-1.0.4.dist-info/METADATA",
                 b"Metadata-Version: 2.1\nName: contest-harness\nVersion: 1.0.4\n", seen)
        add_file(tar, f"{SITE}/contest_harness-1.0.4.dist-info/top_level.txt", b"contest_harness\n", seen)
        with zipfile.ZipFile(io.BytesIO(numpy_wheel())) as wheel:
            for info in wheel.infolist():
                if info.is_dir() or ".data/" in info.filename:
                    continue
                mode = 0o755 if info.filename.endswith(".so") or ".so." in info.filename else 0o644
                add_file(tar, f"{SITE}/{info.filename}", wheel.read(info), seen, mode)
    return buf.getvalue()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("context")
    parser.add_argument("out")
    parser.add_argument("--tag", default="zhixun-rsi:v1")
    args = parser.parse_args(argv)
    context, out = Path(args.context), Path(args.out)
    session = registry()
    index = get_json(session, TAG)
    if "manifests" in index:
        entry = next(m for m in index["manifests"] if m.get("platform", {}).get("architecture") == "amd64"
                     and m.get("platform", {}).get("os") == "linux")
        manifest = get_json(session, entry["digest"])
    else:
        manifest = index
    config = json.loads(blob(session, manifest["config"]["digest"]))
    log(f"base {REPO}:{TAG} amd64, {len(manifest['layers'])} layers")
    work = Path(tempfile.mkdtemp(prefix="image-", dir=out.parent))
    layer_names, diff_ids = [], []
    try:
        for i, layer in enumerate(manifest["layers"]):
            raw = blob(session, layer["digest"])
            tar_bytes = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
            diff = "sha256:" + hashlib.sha256(tar_bytes).hexdigest()
            if diff != config["rootfs"]["diff_ids"][i]:
                raise ValueError(f"base layer {i} diff_id mismatch")
            name = f"{diff[7:]}/layer.tar"
            (work / diff[7:]).mkdir()
            (work / name).write_bytes(tar_bytes)
            layer_names.append(name); diff_ids.append(diff)
            log(f"base layer {i}: {len(tar_bytes) / 1048576:.1f} MB")
        ours = app_layer(context)
        diff = "sha256:" + hashlib.sha256(ours).hexdigest()
        (work / diff[7:]).mkdir()
        (work / f"{diff[7:]}/layer.tar").write_bytes(ours)
        layer_names.append(f"{diff[7:]}/layer.tar"); diff_ids.append(diff)
        log(f"app layer: {len(ours) / 1048576:.1f} MB")

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cfg = config.setdefault("config", {})
        cfg["Env"] = list(cfg.get("Env") or []) + ENV_ADD
        cfg["WorkingDir"] = "/app"
        cfg["User"] = "65534:65534"
        cfg["Entrypoint"] = ["python3", "/app/harness.py"]
        cfg["Cmd"] = None
        cfg.setdefault("Labels", {})
        cfg["Labels"] = dict(cfg["Labels"] or {}, **{"org.opencontainers.image.title": "zhixun-rsi",
                                                       "org.opencontainers.image.version": "v1"})
        config["rootfs"]["diff_ids"] = diff_ids
        config["created"] = now
        config.setdefault("history", []).append({"created": now, "created_by":
            "zhixun-rsi v1: /app + numpy 1.26.4 + contest_harness 1.0.4 (built with build_image_nodocker.py)"})
        config_bytes = json.dumps(config, separators=(",", ":")).encode()
        config_name = hashlib.sha256(config_bytes).hexdigest() + ".json"
        (work / config_name).write_bytes(config_bytes)
        repo, tag = args.tag.split(":")
        (work / "manifest.json").write_text(json.dumps([{"Config": config_name, "RepoTags": [args.tag],
                                                          "Layers": layer_names}]))
        (work / "repositories").write_text(json.dumps({repo: {tag: diff_ids[-1][7:]}}))
        with tarfile.open(out, "w", format=tarfile.PAX_FORMAT) as tar:
            for item in ["manifest.json", "repositories", config_name] + [n.split("/")[0] for n in layer_names]:
                tar.add(work / item, arcname=item)
        log(f"wrote {out} ({out.stat().st_size / 1048576:.1f} MB), config {config_name[:12]}, tag {args.tag}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
