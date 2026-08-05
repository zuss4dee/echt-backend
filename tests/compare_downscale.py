"""
Side-by-side comparison of analysis at full resolution vs. the analysis cap.

The cap (MAX_ANALYSIS_EDGE, see main.max_analysis_edge) exists to stop large
images exhausting memory. Resampling changes high-frequency content, and the
noise detector keys off exactly that — Laplacian variance below 100 raises a
flag that policy treats as critical. So the cap can move verdicts, and this
harness measures whether it does rather than assuming it does not.

Run from backend/:

    ./venv/bin/python tests/compare_downscale.py [files...]

With no arguments it uses tests/_generated plus a synthetic phone photo at the
resolution that was actually OOMing in production.

Reports, per file and per mode: peak RSS, resolved image size, Laplacian
variance, noise flags, verdict and scores. Any row where the verdict or the
noise flag differs between modes is called out explicitly — those are the cases
where the memory fix would have changed a forensic answer.
"""

import os
import resource
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024 / 1024 if sys.platform == "darwin" else kb / 1024


def make_phone_photo(path: str, size: Tuple[int, int] = (4000, 3000)) -> str:
    """A 12MP photo, the shape of input that was being SIGKILLed at 512 MB."""
    from PIL import Image, ImageDraw
    import numpy as np

    rng = np.random.default_rng(1234)
    # Structured content, not pure noise: pure noise has atypically high
    # Laplacian variance and would make the comparison look better than reality.
    base = np.full((size[1], size[0], 3), 240, dtype="uint8")
    base += (rng.normal(0, 6, base.shape)).astype("int16").clip(-40, 40).astype("uint8")
    img = Image.fromarray(base.clip(0, 255).astype("uint8"))
    draw = ImageDraw.Draw(img)
    for i in range(40):
        y = 120 + i * 68
        draw.rectangle([200, y, 200 + (i * 83) % 3200 + 300, y + 26], fill=(35, 35, 40))
    img.save(path, quality=92)
    return path


def analyse(path: str, max_edge: Optional[int]) -> Dict[str, Any]:
    """Run one file through the pipeline in a subprocess-clean environment."""
    # 0 disables the cap. Deleting the variable would NOT do this — the default
    # is 2000, so an unset variable means "capped", and the full-resolution arm
    # would silently measure the capped path against itself.
    os.environ["MAX_ANALYSIS_EDGE"] = "0" if max_edge is None else str(max_edge)

    import main

    with open(path, "rb") as fh:
        data = fh.read()

    before = peak_rss_mb()
    analysis = main.run_full_analysis(os.path.basename(path), data)
    peak = peak_rss_mb()

    doc_key = main.infer_doc_type_key(analysis, "income_evidence")
    # Reuse main's engine: it is the one loaded from config/policies.yaml, and
    # constructing a fresh PolicyEngine from the wrong CWD silently falls back
    # to a near-empty config, which would make this comparison meaningless.
    verdict = main.policy_engine.evaluate(analysis, doc_key).get("verdict", "?")

    img = analysis.get("display_image")
    noise = analysis.get("noise") or {}
    return {
        "size": f"{img.size[0]}x{img.size[1]}" if img is not None else "-",
        "variance": noise.get("variance"),
        "noise_flags": tuple(noise.get("flags") or ()),
        "verdict": verdict,
        "forgery": (analysis.get("metadata") or {}).get("risk_score"),
        "trust": (analysis.get("metadata") or {}).get("trust_score"),
        "peak_mb": peak,
        "delta_mb": peak - before,
    }


def main_cli(argv: List[str]) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    files = argv[1:]
    if not files:
        gen = os.path.join(here, "_generated")
        files = sorted(
            os.path.join(gen, f)
            for f in os.listdir(gen)
            if f.lower().endswith((".pdf", ".jpg", ".jpeg", ".png"))
        )
        photo = os.path.join(here, "_generated", "phone_photo_12mp.jpg")
        if not os.path.exists(photo):
            make_phone_photo(photo)
        if photo not in files:
            files.append(photo)

    mode = os.environ.get("COMPARE_MODE")
    if mode:
        # Child process: one file, one mode, print a parseable line.
        import json

        edge = None if mode == "full" else int(mode)
        print("RESULT" + json.dumps(analyse(files[0], edge), default=str))
        return 0

    # Parent: each (file, mode) runs in its own process so peak RSS is honest
    # and one OOM cannot take the whole comparison down with it.
    import json
    import subprocess

    print(f"\nAnalysis cap comparison — full resolution vs MAX_ANALYSIS_EDGE=2000\n")
    header = (
        f"{'file':<28} {'mode':<6} {'size':>11} {'variance':>10} "
        f"{'verdict':>10} {'peak MB':>8}  noise flag"
    )
    print(header)
    print("-" * (len(header) + 20))

    drift: List[str] = []
    compared = 0
    failures: List[str] = []
    for path in files:
        rows: Dict[str, Optional[Dict[str, Any]]] = {}
        for label, mode_val in (("full", "full"), ("2000", "2000")):
            env = dict(os.environ, COMPARE_MODE=mode_val)
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), path],
                capture_output=True, text=True, env=env,
            )
            line = next(
                (l for l in proc.stdout.splitlines() if l.startswith("RESULT")), None
            )
            if line is None:
                killed = proc.returncode in (-9, 137, 245)
                rows[label] = None
                why = "KILLED (OOM)" if killed else "FAILED"
                err = (proc.stderr or "").strip().splitlines()
                detail = err[-1] if err and not killed else ""
                failures.append(f"{os.path.basename(path)} [{label}] {why} {detail}".strip())
                print(f"{os.path.basename(path):<28} {label:<6} {why:>11}  {detail[:60]}")
                continue
            rows[label] = json.loads(line[len("RESULT"):])

        for label in ("full", "2000"):
            r = rows.get(label)
            if not r:
                continue
            var = r["variance"]
            print(
                f"{os.path.basename(path):<28} {label:<6} {r['size']:>11} "
                f"{(f'{float(var):.2f}' if var is not None else '-'):>10} "
                f"{str(r['verdict']):>10} {float(r['peak_mb']):>8.1f}  "
                f"{'; '.join(r['noise_flags']) or '-'}"
            )

        a, b = rows.get("full"), rows.get("2000")
        if a and b:
            compared += 1
            if a["verdict"] != b["verdict"]:
                drift.append(
                    f"  VERDICT MOVED  {os.path.basename(path)}: "
                    f"{a['verdict']} -> {b['verdict']}"
                )
            elif a["noise_flags"] != b["noise_flags"]:
                drift.append(
                    f"  FLAG CHANGED   {os.path.basename(path)}: "
                    f"{a['noise_flags'] or '()'} -> {b['noise_flags'] or '()'}"
                )
        print()

    if failures:
        print(f"\n{len(failures)} run(s) did not produce a result:")
        for f in failures:
            print(f"  {f}")

    print(f"\nForensic drift introduced by the cap ({compared} file(s) compared):")
    if compared == 0:
        # Never report a clean bill of health from zero comparisons — silence
        # here would mean "nothing ran", not "nothing changed".
        print("  UNKNOWN — no file produced a result in BOTH modes. Nothing was compared.")
        return 1
    print("\n".join(drift) if drift else "  none — every verdict and noise flag identical")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli(sys.argv))
