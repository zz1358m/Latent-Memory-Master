"""
Download compression baseline model checkpoints from HuggingFace.

Models downloaded:
  - microsoft/llmlingua-2-xlm-roberta-large-meetingbank  (LLMLingua-2)
  - princeton-nlp/AutoCompressor-Llama-2-7b-6k           (AutoCompressor)
  - OFA-Sys/xrag-7b                                      (xRAG LLM + projector)
  - facebook/dragon-plus-context-encoder                 (xRAG retrieval)
  - facebook/dragon-plus-query-encoder                   (xRAG retrieval)

All saved under SAVE_ROOT, matching the paths in config.yaml.

Usage:
    python scripts/download_baselines.py
    python scripts/download_baselines.py --save_root models
    python scripts/download_baselines.py --models llmlingua xrag
"""

import argparse
import os
from huggingface_hub import snapshot_download

MODELS = {
    "llmlingua": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "autocompressor": "princeton-nlp/AutoCompressor-Llama-2-7b-6k",
    "xrag": "Hannibal046/xrag-7b",
    "sfr_embedding": "Salesforce/SFR-Embedding-Mistral",
}

def download(repo_id: str, save_root: str):
    local_dir = os.path.join(save_root, repo_id)
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"  already exists, skipping: {local_dir}")
        return
    print(f"  downloading {repo_id} → {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"  done: {local_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_root", default="models",
        help="Root directory under which models are saved (mirrors HF repo structure)"
    )
    parser.add_argument(
        "--models", nargs="*", default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Which models to download (default: all)"
    )
    args = parser.parse_args()

    print(f"Save root: {args.save_root}")
    for key in args.models:
        repo_id = MODELS[key]
        print(f"\n[{key}] {repo_id}")
        download(repo_id, args.save_root)

    print("\nAll downloads complete.")


if __name__ == "__main__":
    main()
