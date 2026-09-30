"""Python 3.7-compatible runner for legacy deep embedding methods.

This file is intentionally standalone: the legacy conda environment cannot
parse parts of the main Python 3.14 figure package.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def method_stem(method):
    return method.lower().replace("-", "").replace(" ", "_")


def sha256_array(arr):
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("utf-8"))
    h.update(str(contiguous.dtype).encode("utf-8"))
    h.update(contiguous.view(np.uint8))
    return h.hexdigest()


def output_path(out_dir, method, seed, tag=None, kind="embedding"):
    suffix = "_seed{}".format(seed) + ("_{}".format(tag) if tag else "")
    stem = method_stem(method)
    if kind == "reconstruction":
        stem = stem + "_reconstruction"
    return Path(out_dir) / "{}{}.npy".format(stem, suffix)


def save_array(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), np.asarray(arr, dtype=np.float32))


def save_meta(path, payload):
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_matrix(args):
    data = np.load(args.input, allow_pickle=False)
    X = data["X"].astype(np.float32)
    if args.max_cells is not None:
        X = X[: args.max_cells, :]
    if args.max_genes is not None:
        X = X[:, : args.max_genes]
    return np.ascontiguousarray(X, dtype=np.float32)


def run_scscope(X, args):
    import tensorflow as tf
    import scscope

    np.random.seed(args.seed)
    try:
        tf.set_random_seed(args.seed)
    except AttributeError:
        tf.compat.v1.set_random_seed(args.seed)

    batch_size = max(1, min(args.batch_size, X.shape[0]))
    started = time.time()
    model = scscope.train(
        X,
        latent_code_dim=args.latent_dim,
        use_mask=True,
        batch_size=batch_size,
        max_epoch=args.scscope_epochs,
        epoch_per_check=max(1, args.scscope_epochs),
        T=args.scscope_recurrence,
        encoder_layers=[],
        decoder_layers=[],
        num_gpus=1,
    )
    latent, reconstruction, _ = scscope.predict(X, model)
    try:
        model["latent_code_session"].close()
    except Exception:
        pass
    return latent, reconstruction, time.time() - started


def saucie_layers(input_dim, latent_dim):
    h1 = min(512, max(16, input_dim // 2))
    h2 = min(256, max(8, h1 // 2))
    h3 = min(128, max(latent_dim * 2, h2 // 2))
    return [int(h1), int(h2), int(h3), int(latent_dim)]


def run_saucie(X, args):
    if args.saucie_parent:
        sys.path.insert(0, args.saucie_parent)
    import tensorflow as tf
    import SAUCIE

    try:
        tf.reset_default_graph()
        tf.set_random_seed(args.seed)
    except AttributeError:
        tf.compat.v1.reset_default_graph()
        tf.compat.v1.set_random_seed(args.seed)
    np.random.seed(args.seed)

    layers = saucie_layers(X.shape[1], args.latent_dim)
    started = time.time()
    model = SAUCIE.SAUCIE(X.shape[1], layers=layers, no_gpu=True)
    loader = SAUCIE.Loader(X, shuffle=True)
    model.train(loader, steps=args.saucie_steps)
    eval_loader = SAUCIE.Loader(X, shuffle=False)
    latent = model.get_embedding(eval_loader)
    reconstruction = model.get_reconstruction(eval_loader)
    try:
        model.sess.close()
    except Exception:
        pass
    return latent, reconstruction, time.time() - started, layers


def run_method(method, X, args):
    normalized = method.lower()
    if normalized == "scscope":
        latent, reconstruction, elapsed = run_scscope(X, args)
        extra = {
            "scscope_epochs": args.scscope_epochs,
            "scscope_recurrence": args.scscope_recurrence,
        }
    elif normalized == "saucie":
        latent, reconstruction, elapsed, layers = run_saucie(X, args)
        extra = {"saucie_steps": args.saucie_steps, "saucie_layers": layers}
    else:
        raise ValueError("Unsupported legacy method: {}".format(method))

    emb_path = output_path(args.out_dir, method, args.seed, args.tag)
    recon_path = output_path(args.out_dir, method, args.seed, args.tag, kind="reconstruction")
    save_array(emb_path, latent)
    save_array(recon_path, reconstruction)
    meta = {
        "method": method,
        "seed": args.seed,
        "tag": args.tag,
        "input": str(args.input),
        "input_shape_used": list(X.shape),
        "input_sha256_used": sha256_array(X),
        "embedding_shape": list(np.asarray(latent).shape),
        "reconstruction_shape": list(np.asarray(reconstruction).shape),
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
    }
    meta.update(extra)
    save_meta(emb_path, meta)
    save_meta(recon_path, meta)
    print("{} OK {} {}".format(method, emb_path, list(np.asarray(latent).shape)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--scscope-epochs", type=int, default=20)
    parser.add_argument("--scscope-recurrence", type=int, default=1)
    parser.add_argument("--saucie-steps", type=int, default=300)
    parser.add_argument("--saucie-parent", default="")
    return parser.parse_args()


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("KMP_AFFINITY", "disabled")
    args = parse_args()
    X = load_matrix(args)
    print("Loaded input", args.input, list(X.shape))
    for method in args.methods:
        run_method(method, X, args)


if __name__ == "__main__":
    main()
