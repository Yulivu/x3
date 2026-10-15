#!/usr/bin/env python3
"""G-C: the counterfactual panel - channel perturbation plus optional vocoder (section 3).

For at most ``--per-platform`` genuine dev clips per platform the package writes

* **channel** variants: the frozen Fam-A/Fam-B chains of ``src.views`` (codec hops,
  band limit, gain, noise), with the library and the query getting *independent*
  chains so a pair can no longer be matched by the encoding itself;
* **vocoder** resynthesis: only when a public vocoder is importable.  The box has no
  network beyond the mirrors, so a missing vocoder is recorded as ``unavailable``
  instead of failing the package (section F1 spirit: never stall the chain).

Everything lands under ``sprint1/counterfactual/`` with a manifest that names every
file, its chain and the clip it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.sprint1 import config as cfg  # noqa: E402
from scripts.sprint1 import state as st  # noqa: E402


def try_vocoder():
    """Return ``(name, fn)`` for the first importable vocoder, else ``(None, None)``."""
    try:  # Vocos ships a pretrained 24 kHz model
        import torch
        from vocos import Vocos

        v = Vocos.from_pretrained("charactr/vocos-mel-24khz")

        def resynth(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(x, np.float32))
                mel = v.feature_extractor(t.unsqueeze(0))
                return v.decode(mel).squeeze().cpu().numpy().astype(np.float32)

        return "vocos-mel-24khz", resynth
    except Exception:
        pass
    try:  # torch.hub needs network; kept as a second chance
        import torch

        vocoder = torch.hub.load("descriptinc/melgan-neurips", "load_melgan", "multi_speaker")

        def resynth_hub(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0)
                return vocoder.infer(t).squeeze().cpu().numpy().astype(np.float32)

        return "melgan-hub", resynth_hub
    except Exception:
        return None, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="X3 sprint 1 G-C counterfactual panel")
    ap.add_argument("--side", default="dev", choices=("dev",))
    ap.add_argument("--per-platform", type=int, default=500)
    ap.add_argument("--out-dir", default="sprint1/counterfactual")
    ap.add_argument("--max-minutes", type=float, default=80.0)
    ap.add_argument("--chain", default="fam_a", choices=("fam_a", "fam_b"))
    ap.add_argument("--no-vocoder", action="store_true")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else cfg.REPO_ROOT
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else root / out_dir
    (out_dir / "channel").mkdir(parents=True, exist_ok=True)
    index_file = root / "sprint1/audio/dev_index.json"
    if not index_file.exists():
        print(f"missing {index_file}: run P-1 first")
        return 2
    audio_index = json.loads(index_file.read_text(encoding="utf-8"))
    if audio_index.get("side") != "dev":
        print("REFUSED: only the dev memmap may be used before FREEZE_OK")
        return 3
    mm = np.memmap(audio_index["path"], dtype=audio_index["dtype"], mode="r",
                   shape=tuple(audio_index["shape"]))
    meta = audio_index["meta"]
    rows = np.asarray(audio_index["rows"], int)
    y = np.asarray(meta["y"], int)
    plat = np.asarray(meta["platform"]).astype(str)

    from src.views import fam_a_params, fam_b_params, apply_view, write_wav_mono

    gen = np.random.default_rng(cfg.DRAW_BASE + 77)
    picked: List[int] = []
    for p in sorted(set(plat)):
        idx = np.where((plat == p) & (y == 0))[0]
        if idx.size == 0:
            continue
        take = min(int(args.per_platform), idx.size)
        picked.extend(int(v) for v in gen.choice(idx, size=take, replace=False))
    picked = sorted(set(picked))

    t0 = time.time()
    entries: List[Dict[str, Any]] = []
    voc_name = voc_fn = None
    if not args.no_vocoder:
        (out_dir / "vocoder").mkdir(exist_ok=True)
        voc_name, voc_fn = try_vocoder()
    for k, i in enumerate(picked):
        x = np.asarray(mm[i], np.float32) / 32767.0
        clip_id = int(audio_index["clip_id"][i])
        params = (fam_a_params(clip_id) if args.chain == "fam_a"
                  else fam_b_params(clip_id, "B1"))
        try:
            yv = apply_view(x, params, clip_id)
            path = out_dir / "channel" / f"{clip_id}_{args.chain}.wav"
            write_wav_mono(path, yv)
            entry = {"clip_id": clip_id, "row": int(rows[i]), "platform": str(plat[i]),
                     "label": int(y[i]), "chain": args.chain,
                     "channel_path": str(path.relative_to(root)), "params": params}
        except Exception as exc:                                  # noqa: BLE001 - reported
            entry = {"clip_id": clip_id, "row": int(rows[i]), "platform": str(plat[i]),
                     "chain": args.chain, "channel_error": f"{type(exc).__name__}: {exc}"}
        if voc_fn is not None:
            try:
                vpath = out_dir / "vocoder" / f"{clip_id}.wav"
                write_wav_mono(vpath, voc_fn(x))
                entry["vocoder_path"] = str(vpath.relative_to(root))
                entry["vocoder"] = voc_name
            except Exception as exc:                              # noqa: BLE001 - reported
                entry["vocoder_error"] = f"{type(exc).__name__}: {exc}"
        entries.append(entry)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(picked)} clips", flush=True)
        if time.time() - t0 > args.max_minutes * 60:
            print("time cap reached; stopping the panel here", flush=True)
            break

    manifest = {
        "created": st.now_iso(), "side": "dev", "chain": args.chain,
        "vocoder": voc_name or "unavailable",
        "n_selected": len(picked), "n_written": len(entries),
        "per_platform_cap": int(args.per_platform),
        "seconds": round(time.time() - t0, 1),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                      default=str), encoding="utf-8")
    print(json.dumps({"n_written": len(entries), "vocoder": manifest["vocoder"],
                      "seconds": manifest["seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
