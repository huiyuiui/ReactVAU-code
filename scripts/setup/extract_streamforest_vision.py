#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Extract ReactVAU's shared vision encoder from a StreamForest checkpoint."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_PREFIX = "model.vision_tower.vision_tower."
EXPECTED_SHAPES = {
    "vision_model.embeddings.patch_embedding.weight": (1152, 3, 14, 14),
    "vision_model.embeddings.position_embedding.weight": (729, 1152),
    "vision_model.post_layernorm.weight": (1152,),
}


def extract_vision_encoder(streamforest_dir: Path, output_path: Path) -> int:
    index_path = streamforest_dir / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file)["weight_map"]

    source_keys_by_shard = defaultdict(list)
    for source_key, shard_name in weight_map.items():
        if source_key.startswith(SOURCE_PREFIX + "vision_model."):
            source_keys_by_shard[shard_name].append(source_key)

    tensor_count = sum(len(keys) for keys in source_keys_by_shard.values())
    if tensor_count != 421:
        raise ValueError(
            f"expected 421 vision tensors, found {tensor_count}; "
            "check that --streamforest-dir points to the official "
            "MCG-NJU/StreamForest-Qwen2-7B checkpoint"
        )

    extracted = {}
    for shard_name, source_keys in sorted(source_keys_by_shard.items()):
        with safe_open(
            streamforest_dir / shard_name, framework="pt", device="cpu"
        ) as shard:
            for source_key in sorted(source_keys):
                output_key = source_key.removeprefix(SOURCE_PREFIX)
                extracted[output_key] = shard.get_tensor(source_key)

    for key, expected_shape in EXPECTED_SHAPES.items():
        if key not in extracted:
            raise ValueError(f"required tensor is missing: {key}")
        actual_shape = tuple(extracted[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"{key}: expected {expected_shape}, found {actual_shape}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(extracted, output_path)
    return len(extracted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the shared SigLIP-384 vision encoder required by ReactVAU "
            "from the official StreamForest checkpoint."
        )
    )
    parser.add_argument(
        "--streamforest-dir",
        type=Path,
        default=Path("ckpt/StreamForest-Qwen2-7B_Siglip"),
        help="Directory containing the downloaded StreamForest checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ckpt/extracted_weights/streamforest_vision_encoder_with_prefix.safetensors"
        ),
        help="Output safetensors path.",
    )
    args = parser.parse_args()

    tensor_count = extract_vision_encoder(args.streamforest_dir, args.output)
    print(f"Extracted {tensor_count} tensors to {args.output}")


if __name__ == "__main__":
    main()
