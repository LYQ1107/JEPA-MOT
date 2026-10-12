#!/usr/bin/env python3
"""Render the six selected tracklets at their final single-column size."""

from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import plot_other_datasets_tracklet_comparisons as other
import plot_vkitti2_tracklet_comparisons as vk


OUTPUT = ROOT / "work_dirs/visualizations/selected_six_tracklets"
COLORS = vk.COLORS

SPECS = [
    ("motsynth", "ctaod", "dark", "dark/050", 20, 91, "(a)", "dark/050"),
    ("motsynth", "ctaod", "fog", "fog/530", 56, 4234, "(b)", "fog/530"),
    ("motsynth", "ctaod", "fog", "fog/515", 35, 3556, "(c)", "fog/515"),
    (
        "vkitti2",
        "darth",
        "rain",
        "Scene01_rain_Camera_1",
        5,
        463,
        "(d)",
        "S01 rain",
    ),
    (
        "vkitti2",
        "darth",
        "sunset",
        "Scene20_sunset_Camera_1",
        11,
        3550,
        "(e)",
        "S20 sunset",
    ),
    (
        "vkitti2",
        "darth",
        "morning",
        "Scene01_morning_Camera_1",
        9,
        473,
        "(f)",
        "S01 morning",
    ),
]


def select_exact(candidates, video_name, gt_id, frame_start):
    matches = [
        item
        for item in candidates
        if item.video_name == video_name
        and item.gt_id == gt_id
        and item.gt[0].frame == frame_start
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one candidate for {video_name}, ID {gt_id}, "
            f"frame {frame_start}; found {len(matches)}"
        )
    return matches[0]


def load_motsynth(domain):
    config = other.CONFIGS["motsynth"]
    output_dir = (
        ROOT / "work_dirs/visualizations/other_datasets_tracklets/motsynth"
    )
    annotation = ROOT / config.annotation_pattern.format(domain=domain)
    ours = vk.load_coco_tracks(other.find_single(config.ours_patterns[domain]))
    ctaod = other.rebuild_ctta_tracks(config, domain, output_dir)
    return other.build_candidates(
        "ctaod",
        domain,
        annotation,
        ours,
        ctaod,
        config.min_movement,
        30,
        0.20,
    )


def load_vkitti2(domain):
    ours = vk.load_coco_tracks(vk.ours_track_path(domain))
    darth = vk.load_coco_tracks(vk.darth_track_path(domain))
    return vk.build_candidates("darth", domain, ours, darth, 0.25, 30, 150.0, 30)


def plot_candidate(ax, candidate, panel, short_sequence, dataset, comparison):
    gt = np.asarray([item.center for item in candidate.gt])
    ax.plot(
        gt[:, 0],
        gt[:, 1],
        color=COLORS["gt"],
        linewidth=0.55,
        marker="x",
        markersize=2.3,
        markeredgewidth=0.55,
        alpha=0.9,
        zorder=1,
    )
    for points in vk.segments(candidate.comp):
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS[comparison],
            linewidth=0.65,
            marker="s",
            markersize=2.2,
            markeredgewidth=0,
            alpha=0.82,
            zorder=2,
        )
    for points in vk.segments(candidate.ours):
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS["ours"],
            linewidth=0.85,
            marker="o",
            markersize=2.7,
            markeredgecolor="white",
            markeredgewidth=0.25,
            alpha=0.98,
            zorder=3,
        )

    arrays = [gt, *vk.segments(candidate.ours), *vk.segments(candidate.comp)]
    merged = np.concatenate([item for item in arrays if len(item)], axis=0)
    x_margin = max(25.0, 0.08 * np.ptp(merged[:, 0]))
    y_margin = max(18.0, 0.10 * np.ptp(merged[:, 1]))
    ax.set_xlim(merged[:, 0].min() - x_margin, merged[:, 0].max() + x_margin)
    ax.set_ylim(merged[:, 1].max() + y_margin, merged[:, 1].min() - y_margin)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#3d3d3d")
        spine.set_linewidth(0.45)

    frame_range = f"{candidate.gt[0].frame}-{candidate.gt[-1].frame}"
    ax.set_title(
        f"{panel} {short_sequence}\nID {candidate.gt_id} | F{frame_range}",
        fontsize=7.2,
        fontweight="semibold",
        pad=2.2,
        linespacing=1.0,
    )


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    loaded = {}
    selected = []
    for dataset, comparison, domain, video, gt_id, start, panel, short in SPECS:
        key = (dataset, domain)
        if key not in loaded:
            loaded[key] = (
                load_motsynth(domain)
                if dataset == "motsynth"
                else load_vkitti2(domain)
            )
        selected.append(
            (
                select_exact(loaded[key], video, gt_id, start),
                panel,
                short,
                "MOTSynth" if dataset == "motsynth" else "VKITTI2",
                comparison,
            )
        )

    png_path = OUTPUT / "combined_selected_six_tracklets.png"
    pdf_path = OUTPUT / "combined_selected_six_tracklets.pdf"
    if png_path.exists():
        shutil.copy2(png_path, OUTPUT / "combined_selected_six_tracklets_large.png")
    if pdf_path.exists():
        shutil.copy2(pdf_path, OUTPUT / "combined_selected_six_tracklets_large.pdf")

    fig, axes = plt.subplots(2, 3, figsize=(3.45, 2.58))
    for ax, values in zip(axes.flat, selected):
        plot_candidate(ax, *values)

    legend_handles = [
        mlines.Line2D(
            [], [], color=COLORS["ours"], marker="o", markersize=3.0,
            linewidth=0.9, label="Ours"
        ),
        mlines.Line2D(
            [], [], color=COLORS["ctaod"], marker="s", markersize=2.5,
            linewidth=0.7, label="Baseline"
        ),
        mlines.Line2D(
            [], [], color=COLORS["gt"], marker="x", markersize=2.7,
            linewidth=0.6, label="Ground truth"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.922),
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.8,
        columnspacing=1.25,
    )
    fig.text(
        0.5,
        0.987,
        "(a-c) MOTSynth: Ours vs CTAOD    |    "
        "(d-f) VKITTI2: Ours vs DARTH",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="semibold",
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.995,
        bottom=0.025,
        top=0.775,
        wspace=0.12,
        hspace=0.34,
    )
    fig.savefig(png_path, dpi=800, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
