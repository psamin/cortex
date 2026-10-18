"""Composite the parallel-RL run (rl_grid.py output) into one 1920x1080 H.264 video.

    python render_video.py <out_dir>

A 10x5 grid of the 50 rollouts of each CEM iteration playing at once (border: grey running, green success,
red failure), a header saying what runs where, and a footer with that iteration's measured success and a
success-per-iteration bar chart. Title and summary cards use only numbers from the run.
"""

import io
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 15
TW, TH, COLS = 192, 144, 10
GRID_Y = 132
BG, GREY, GREEN, RED, WHITE, BLUE = (14, 14, 14), (90, 90, 90), (90, 210, 120), (220, 80, 80), (240, 240, 240), (160, 200, 255)


def font(n):
    try:
        return ImageFont.load_default(size=n)
    except TypeError:
        return ImageFont.load_default()


F_BIG, F_MID, F_SM = font(40), font(26), font(20)


def card(lines):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = H // 2 - 40 * len(lines) // 2
    for text, f, color in lines:
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, font=f, fill=color)
        y += 64 if f is F_BIG else 44
    return np.asarray(im)


def main():
    out = Path(sys.argv[1])
    meta = json.loads((out / "meta.json").read_text())
    stats = [json.loads((out / f"iter{i}.json").read_text()) for i in range(1, meta["iterations"] + 1)]
    pop = meta["population"]
    header1 = "Cortex RL sandbox: cross-entropy method (CEM) tuning the pick-and-place skill"
    header2 = (f"{pop} rollouts in parallel per iteration  ·  MuJoCo physics on {meta['processes']} CPU cores  ·  "
               f"rendered on 1x {meta['gpu']}")
    c = av.open(str(out / "rl_parallel.mp4"), "w")
    s = c.add_stream("libx264", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.options = {"crf": "21", "preset": "veryfast", "movflags": "+faststart"}

    def emit(arr, n=1):
        for _ in range(n):
            for p in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)

    first, last = stats[0]["success"] / pop, stats[-1]["success"] / pop
    emit(card([("Massively parallel RL in simulation", F_BIG, WHITE),
               (f"CEM searches the skill's grasp / release corrections: {pop} rollouts at once, {meta['iterations']} iterations",
                F_MID, BLUE),
               ("It starts from a deliberately mis-calibrated skill (grasp ~1.5 cm off, jaws turned 16 degrees)", F_MID, BLUE)]),
         int(FPS * 2.5))
    for k, st in enumerate(stats):
        roll = np.load(out / f"iter{k + 1}.npy", allow_pickle=True)
        tiles = [[np.asarray(Image.open(io.BytesIO(j)).convert("RGB").resize((TW - 6, TH - 6))) for j in r["frames"]]
                 for r in roll]
        n = max(len(t) for t in tiles)
        for f in range(n + FPS):  # one extra second on the final state
            im = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(im)
            d.text((24, 18), header1, font=F_MID, fill=WHITE)
            d.text((24, 62), header2, font=F_SM, fill=BLUE)
            done_ok = 0
            for i, (t, r) in enumerate(zip(tiles, roll)):
                x, y = (i % COLS) * TW, GRID_Y + (i // COLS) * TH
                ended = f >= len(t) - 1
                color = (GREEN if r["ok"] else RED) if ended else GREY
                done_ok += int(ended and r["ok"])
                d.rectangle([x, y, x + TW - 1, y + TH - 1], fill=color)
                im.paste(Image.fromarray(t[min(f, len(t) - 1)]), (x + 3, y + 3))
            fy = GRID_Y + 5 * TH + 22
            d.text((24, fy), f"iteration {st['iteration']}/{meta['iterations']}", font=F_BIG, fill=WHITE)
            d.text((24, fy + 58), f"succeeded so far this iteration: {done_ok}/{pop}", font=F_MID, fill=GREEN)
            d.text((24, fy + 98), f"final: {st['success']}/{pop} successful  ·  mean score {st['mean_score']:+.3f}"
                   if f >= n - 1 else "rollouts running…", font=F_MID, fill=BLUE)
            d.text((24, fy + 138), "score = 1 if the cube ends released in the box, minus its distance to the box (m)",
                   font=F_SM, fill=GREY)
            bx, by, bw, bh = 1180, fy + 8, 700, 170  # success rate per iteration
            d.text((bx, by - 4), "success rate per iteration", font=F_SM, fill=WHITE)
            for j, sj in enumerate(stats):
                if j > k or (j == k and f < n - 1):
                    continue
                rate = sj["success"] / pop
                x0 = bx + j * (bw // len(stats)) + 8
                x1 = x0 + bw // len(stats) - 16
                d.rectangle([x0, by + bh - int(rate * (bh - 30)), x1, by + bh], fill=GREEN)
                d.text((x0, by + bh - int(rate * (bh - 30)) - 24), f"{rate:.0%}", font=F_SM, fill=WHITE)
            emit(np.asarray(im))
    emit(card([("Result (measured)", F_BIG, WHITE),
               (f"success {first:.0%} -> {last:.0%} over {meta['iterations']} CEM iterations", F_BIG, GREEN),
               (f"{pop * meta['iterations']} rollouts, {pop} at a time  ·  RL wall time {meta['rl_wall_s']:.0f} s on "
                f"{meta['processes']} CPU cores + 1x {meta['gpu']}", F_MID, BLUE),
               ("The tuned skill's successful rollouts become targeted demos for retraining the learned policy", F_MID, BLUE)]),
         FPS * 4)
    for p in s.encode():
        c.mux(p)
    c.close()
    print("[video] DONE", out / "rl_parallel.mp4", flush=True)


if __name__ == "__main__":
    main()
