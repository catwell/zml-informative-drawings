#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "torch",
#     "safetensors",
#     "packaging",
#     "numpy",
#     "httpx",
# ]
# ///
"""Convert Informative Drawings PyTorch weights to safetensors format."""

import tempfile
from pathlib import Path

import click
import httpx
import torch
from safetensors.torch import save_file

DEFAULT_URL = (
    "https://huggingface.co/spaces/carolineec/informativedrawings"
    "/resolve/main/model2.pth"
)


def download(url: str, dest: Path) -> None:
    click.echo(f"Downloading {url} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with dest.open("wb") as f, click.progressbar(length=total, label="Progress") as bar:
            for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                bar.update(len(chunk))


@click.command()
@click.option(
    "--from",
    "src",
    default=DEFAULT_URL,
    show_default=True,
    help="Path or URL to the input .pth file.",
)
@click.option(
    "--to",
    "dst",
    default="model2.safetensors",
    show_default=True,
    help="Output path for the .safetensors file.",
)
def main(src: str, dst: str) -> None:
    """Convert Informative Drawings PyTorch weights to safetensors format."""
    dst_path = Path(dst)

    if src.startswith(("http://", "https://")):
        tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        download(src, tmp_path)
        input_path = tmp_path
    else:
        input_path = Path(src)

    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)
    save_file(state_dict, str(dst_path))
    click.echo(f"Converted {src} -> {dst_path}")
    click.echo(f"Keys: {list(state_dict.keys())}")


if __name__ == "__main__":
    main()
