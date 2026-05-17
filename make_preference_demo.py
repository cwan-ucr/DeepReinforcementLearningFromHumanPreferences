from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from sumo_rlhf.preference_data import load_preference_labels
from sumo_rlhf.trajectory_buffer import TrajectoryBuffer
from sumo_rlhf.trajectory_plot import plot_segment_pair


PREFERENCE_TEXT = {
    0: "Left Preferred",
    1: "Right Preferred",
    2: "Neutral",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a short demo animation from preference labeling history."
    )
    parser.add_argument("--segments", required=True, help="Trajectory pool jsonl used for labeling.")
    parser.add_argument("--preferences", required=True, help="Preference labels jsonl in labeling order.")
    parser.add_argument("--output", default="runs/preference_labeling_demo.gif")
    parser.add_argument("--plot-dir", default="runs/preference_demo_frames")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--seconds-per-label",
        type=float,
        default=1.5,
        help="Hold time for each labeled pair in the demo animation.",
    )
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip labels whose segment ids do not exist in --segments.",
    )
    return parser.parse_args()


def render_pair_frames(
    segments_path: str,
    preferences_path: str,
    plot_dir: Path,
    max_pairs: int,
    skip_missing: bool,
):
    buffer = TrajectoryBuffer.load_jsonl(segments_path)
    segment_by_id = {segment.segment_id: segment for segment in buffer.segments}
    labels = load_preference_labels(preferences_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    frames: list[tuple[Path, str]] = []
    for idx, label in enumerate(labels[:max_pairs], start=1):
        left = segment_by_id.get(label.left_id)
        right = segment_by_id.get(label.right_id)
        if left is None or right is None:
            if skip_missing:
                continue
            missing_ids = []
            if left is None:
                missing_ids.append(label.left_id)
            if right is None:
                missing_ids.append(label.right_id)
            raise KeyError(
                f"missing segment ids in pool: {', '.join(missing_ids)} for label index {idx}"
            )

        frame_path = plot_dir / f"label_{idx:04d}.png"
        plot_segment_pair(left, right, frame_path)
        decision = PREFERENCE_TEXT.get(int(label.preference), str(label.preference))
        title = f"Label {idx}: {decision}"
        frames.append((frame_path, title))
    return frames


def save_animation(
    frames: list[tuple[Path, str]],
    output_path: Path,
    fps: int,
    seconds_per_label: float,
):
    if not frames:
        raise RuntimeError("No frames were generated. Check labels and segment ids.")

    hold_frames = max(1, int(round(seconds_per_label * fps)))
    first_img = plt.imread(frames[0][0])
    height, width = first_img.shape[0], first_img.shape[1]
    fig_w = max(8.0, width / 180.0)
    fig_h = max(4.5, height / 180.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    ax.axis("off")

    writer = (
        PillowWriter(fps=fps)
        if output_path.suffix.lower() == ".gif"
        else FFMpegWriter(fps=fps)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(output_path), dpi=180):
        for frame_path, title in frames:
            image = plt.imread(frame_path)
            for _ in range(hold_frames):
                ax.clear()
                ax.axis("off")
                ax.imshow(image)
                ax.set_title(title, fontsize=14, pad=8)
                writer.grab_frame()
    plt.close(fig)


def main():
    args = parse_args()
    output_path = Path(args.output)
    plot_dir = Path(args.plot_dir)
    frames = render_pair_frames(
        args.segments,
        args.preferences,
        plot_dir=plot_dir,
        max_pairs=args.max_pairs,
        skip_missing=args.skip_missing,
    )
    save_animation(
        frames,
        output_path=output_path,
        fps=args.fps,
        seconds_per_label=args.seconds_per_label,
    )
    print(f"saved preference labeling demo to {output_path} with {len(frames)} labeled pairs")


if __name__ == "__main__":
    main()
