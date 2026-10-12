#!/usr/bin/env python3
"""Compose selected MOTSynth and VKITTI2 tracklet figures."""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "work_dirs/visualizations/selected_six_tracklets"

ITEMS = [
    {
        "dataset": "MOTSynth",
        "comparison": "CTAOD (ALL_5_BELOW)",
        "sequence": "dark/050",
        "gt_id": 20,
        "frames": "91-120",
        "source": ROOT / "work_dirs/visualizations/other_datasets_tracklets/motsynth/figures/tracklet_07_ctaod_dark",
    },
    {
        "dataset": "MOTSynth",
        "comparison": "CTAOD (ALL_5_BELOW)",
        "sequence": "fog/530",
        "gt_id": 56,
        "frames": "4234-4263",
        "source": ROOT / "work_dirs/visualizations/other_datasets_tracklets/motsynth/figures/tracklet_09_ctaod_fog",
    },
    {
        "dataset": "MOTSynth",
        "comparison": "CTAOD (ALL_5_BELOW)",
        "sequence": "fog/515",
        "gt_id": 35,
        "frames": "3556-3585",
        "source": ROOT / "work_dirs/visualizations/other_datasets_tracklets/motsynth/figures/tracklet_12_ctaod_fog",
    },
    {
        "dataset": "VKITTI2",
        "comparison": "DARTH",
        "sequence": "Scene01_rain_Camera_1",
        "gt_id": 5,
        "frames": "463-492",
        "source": ROOT / "work_dirs/visualizations/vkitti2_tracklet_comparison/figures/tracklet_03_darth_rain",
    },
    {
        "dataset": "VKITTI2",
        "comparison": "DARTH",
        "sequence": "Scene20_sunset_Camera_1",
        "gt_id": 11,
        "frames": "3550-3579",
        "source": ROOT / "work_dirs/visualizations/vkitti2_tracklet_comparison/figures/tracklet_01_darth_sunset",
    },
    {
        "dataset": "VKITTI2",
        "comparison": "DARTH",
        "sequence": "Scene01_morning_Camera_1",
        "gt_id": 9,
        "frames": "473-502",
        "source": ROOT / "work_dirs/visualizations/vkitti2_tracklet_comparison/figures/tracklet_02_darth_morning",
    },
]


def safe_name(index, item):
    sequence = item["sequence"].replace("/", "_")
    comparison = item["comparison"].split()[0].lower()
    return (
        f"{index:02d}_{item['dataset'].lower()}_ours_vs_{comparison}_"
        f"{sequence}_gt{item['gt_id']}_frames{item['frames']}"
    )


def font(size, bold=False):
    suffix = "TIMESBD.TTF" if bold else "TIMES.TTF"
    return ImageFont.truetype(
        f"/home/lwr/.local/share/fonts/TimesNewRoman/{suffix}", size
    )


def annotate(item):
    source = Image.open(item["source"].with_suffix(".png")).convert("RGB")
    banner_height = 370
    output = Image.new("RGB", (source.width, source.height + banner_height), "white")
    output.paste(source, (0, banner_height))
    draw = ImageDraw.Draw(output)
    sequence = item["sequence"].replace("_", " ")
    title = f"{item['dataset']} | Ours vs {item['comparison']}"
    details = f"{sequence} | GT ID {item['gt_id']} | Frames {item['frames']}"
    draw.text((90, 40), title, fill="#111111", font=font(108, bold=True))
    draw.text((90, 195), details, fill="#333333", font=font(88))
    draw.line((90, 335, source.width - 90, 335), fill="#c7c7c7", width=3)
    return output


def save_pdf(image, path):
    fig = plt.figure(figsize=(image.width / 300, image.height / 300), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.axis("off")
    fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    annotated = []
    for index, item in enumerate(ITEMS, 1):
        image = annotate(item)
        name = safe_name(index, item)
        png_path = OUTPUT / f"{name}.png"
        pdf_path = OUTPUT / f"{name}.pdf"
        image.save(png_path, optimize=True)
        save_pdf(image, pdf_path)
        shutil.copy2(item["source"].with_suffix(".pdf"), OUTPUT / f"{name}_vector.pdf")
        annotated.append(image)

    panel_width = 2800
    resized = []
    for image in annotated:
        panel_height = round(image.height * panel_width / image.width)
        resized.append(image.resize((panel_width, panel_height), Image.Resampling.LANCZOS))

    gutter = 50
    row_height = max(image.height for image in resized)
    canvas = Image.new(
        "RGB",
        (panel_width * 3 + gutter * 4, row_height * 2 + gutter * 3),
        "#ededed",
    )
    for index, image in enumerate(resized):
        row, column = divmod(index, 3)
        x = gutter + column * (panel_width + gutter)
        y = gutter + row * (row_height + gutter)
        canvas.paste(image, (x, y))

    combined_png = OUTPUT / "combined_selected_six_tracklets.png"
    combined_pdf = OUTPUT / "combined_selected_six_tracklets.pdf"
    canvas.save(combined_png, optimize=True)
    save_pdf(canvas, combined_pdf)


if __name__ == "__main__":
    main()
