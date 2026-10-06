from __future__ import annotations

import argparse
import pickle
import tarfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import timm
import torch
from PIL import Image, ImageFile
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
}


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure PIL image is RGB; convert RGBA to RGB on white."""

    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    """Pad image to square using white background."""

    width, height = image.size
    edge = max(image.size)
    canvas = Image.new("RGB", (edge, edge), (255, 255, 255))
    canvas.paste(image, ((edge - width) // 2, (edge - height) // 2))
    return canvas


def iter_tar_images(tar_path: Path) -> Iterator[Tuple[str, Image.Image]]:
    """Yield image name and padded RGB image for each valid image file inside a tar archive."""

    with tarfile.open(tar_path, mode="r") as handle:
        for member in handle:
            if not member.isfile():
                continue
            member_name = member.name
            lower_name = member_name.lower()
            if not lower_name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                continue
            extract_file = handle.extractfile(member)
            if extract_file is None:
                continue
            try:
                with Image.open(extract_file) as pil_image:
                    pil_image.load()
                    rgb_image = pil_ensure_rgb(pil_image)
                    yield member_name, pil_pad_square(rgb_image)
            except (OSError, ValueError):
                continue


def build_model(repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> nn.Module:
    """Instantiate tagger model and load weights from Hugging Face hub."""

    model: nn.Module = timm.create_model(f"hf-hub:{repo_id}").eval()
    state_dict = timm.models.load_state_dict_from_hf(repo_id, revision=revision, token=token)
    model.load_state_dict(state_dict)
    return model


def prepare_transform(model: nn.Module):
    """Create preprocessing transform from model config."""

    return create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


def to_bgr(tensor: Tensor) -> Tensor:
    """Convert tensor from RGB to BGR channel order."""

    return tensor[[2, 1, 0], ...]


def load_selected_indices(csv_path: Path) -> List[int]:
    """Load index_in_full column to select subset of logits."""

    frame = pd.read_csv(csv_path, usecols=["index_in_full"])
    return frame["index_in_full"].astype(int).tolist()


@dataclass
class EncodeOptions:
    tar_dir: Path = Path("/nfs/danbooru-wdtagger-v4-w640-ws-30k/train")
    tags_map_path: Path = Path("/nfs/danbooru-wdtagger-v4-w640-ws-30k/tags_map.csv")
    output_path: Path = Path("/nfs/danbooru-wdtagger-v4-w640-ws-30k/train/train_part_probs.pkl")
    model: str = "vit"
    batch_size: int = 32
    revision: Optional[str] = None
    token: Optional[str] = None


def parse_args() -> EncodeOptions:
    parser = argparse.ArgumentParser(description="Encode Danbooru tar archives into tag probabilities.")
    parser.add_argument("--tar-dir", type=Path, default=EncodeOptions.tar_dir, help="Directory containing tar archives.")
    parser.add_argument(
        "--tags-map-path",
        type=Path,
        default=EncodeOptions.tags_map_path,
        help="CSV with index_in_full column used for selection.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=EncodeOptions.output_path,
        help="Destination pickle file for part probabilities.",
    )
    parser.add_argument("--model", type=str, choices=list(MODEL_REPO_MAP.keys()), default=EncodeOptions.model)
    parser.add_argument("--batch-size", type=int, default=EncodeOptions.batch_size)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()
    return EncodeOptions(
        tar_dir=args.tar_dir,
        tags_map_path=args.tags_map_path,
        output_path=args.output_path,
        model=args.model,
        batch_size=args.batch_size,
        revision=args.revision,
        token=args.token,
    )


def run_batch(model: nn.Module, batch_tensors: List[Tensor]) -> Tensor:
    inputs = torch.stack(batch_tensors, dim=0)
    if torch_device.type != "cpu":
        inputs = inputs.to(torch_device, non_blocking=True)
    with torch.cuda.amp.autocast(dtype=torch.float16) if torch_device.type == "cuda" else nullcontext():
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
    return outputs.to("cpu", dtype=torch.float32)


def store_outputs(
    results: Dict[str, Tensor],
    image_names: List[str],
    outputs: Tensor,
    selected_indices: List[int],
) -> None:
    subset = outputs[:, selected_indices]
    for name, tensor in zip(image_names, subset):
        results[name] = tensor.clone()


def encode_archives(opts: EncodeOptions) -> Dict[str, Tensor]:
    repo_id = MODEL_REPO_MAP.get(opts.model)
    if repo_id is None:
        raise ValueError(f"Unknown model '{opts.model}'. Available: {list(MODEL_REPO_MAP.keys())}")

    model = build_model(repo_id, revision=opts.revision, token=opts.token)
    model.to(torch_device)
    transform = prepare_transform(model)
    selected_indices = load_selected_indices(opts.tags_map_path)
    results: Dict[str, Tensor] = {}

    tar_paths = sorted(path for path in opts.tar_dir.glob("*.tar"))
    with torch.inference_mode():
        for tar_path in tqdm(tar_paths, desc="Archives", unit="tar"):
            image_names: List[str] = []
            batch_tensors: List[Tensor] = []
            for name, image in iter_tar_images(tar_path):
                tensor = transform(image)
                image.close()
                tensor = to_bgr(tensor)
                batch_tensors.append(tensor)
                image_names.append(Path(name).stem)
                if len(batch_tensors) == opts.batch_size:
                    outputs = run_batch(model, batch_tensors)
                    store_outputs(results, image_names, outputs, selected_indices)
                    batch_tensors.clear()
                    image_names.clear()
            if batch_tensors:
                outputs = run_batch(model, batch_tensors)
                store_outputs(results, image_names, outputs, selected_indices)
    return results


def save_results(results: Dict[str, Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(results, handle)


def main() -> None:
    options = parse_args()
    results = encode_archives(options)
    save_results(results, options.output_path)


if __name__ == "__main__":
    main()

