#!/usr/bin/env python3
"""Generate the animated Mpemba-mechanism summary used in the README.

The animation is deliberately schematic.  The two monotone curves form a
near-equilibrium, two-mode example with an exact crossing:

    M_A(t) = (1/12) exp(-0.30 t) + (11/12) exp(-0.60 t)
    M_B(t) = (1/3)  exp(-0.30 t) + (5/12)  exp(-0.60 t)

Thus A starts more resourceful, while its weight in the slow mode is four
times smaller.  The curves cross once at log(2) / 0.30.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


WIDTH = 960
HEIGHT = 400
SCALE = 2
FPS = 12
DURATION = 8.0

BACKGROUND = "#FFFFFF"
INK = "#000000"
MUTED = "#595959"
STRUCTURE = "#898989"
GRID = "#DFDFDF"
PANEL = "#F7F7F7"
PANEL_ALT = "#F7F7F7"
STATE_A_COLOR = "#000000"
STATE_B_COLOR = "#000000"
GOLD = "#F2C865"
GOLD_PALE = "#FFF8EB"
GOLD_DARK = "#BA903E"
SLOW_MODE_COLOR = "#F2C865"
FREE_SET_COLOR = "#898989"
WHITE = "#FFFFFF"
DESIGN_COLOR_HEXES = (
    BACKGROUND,
    INK,
    MUTED,
    STRUCTURE,
    GRID,
    PANEL,
    GOLD_PALE,
    GOLD,
    GOLD_DARK,
)

SLOW_DECAY = 0.30
FAST_DECAY = 0.60
A_SLOW_WEIGHT = 1.0 / 12.0
A_FAST_WEIGHT = 11.0 / 12.0
B_SLOW_WEIGHT = 1.0 / 3.0
B_FAST_WEIGHT = 5.0 / 12.0
CROSSING_TIME = math.log(2.0) / SLOW_DECAY
CROSSING_VALUE = 13.0 / 48.0

_MATPLOTLIB_SPEC = find_spec("matplotlib")
MATPLOTLIB_FONT_DIR = (
    Path(_MATPLOTLIB_SPEC.origin).parent / "mpl-data" / "fonts" / "ttf"
    if _MATPLOTLIB_SPEC is not None and _MATPLOTLIB_SPEC.origin is not None
    else None
)

REFERENCE_FIGURE = (
    Path(__file__).resolve().parents[1]
    / "figures"
    / "figure_1_resource_theory_mpemba.png"
)
ICON_SOURCE_WINDOWS = {
    "athermality": (106, 77, 165, 140),
    "asymmetry": (103, 194, 173, 252),
    "nonstationarity": (214, 109, 269, 166),
    "coherence": (224, 183, 279, 238),
}


with Image.open(REFERENCE_FIGURE) as reference_image:
    REFERENCE_COLOR_COUNTS = Counter(reference_image.convert("RGB").getdata())
REFERENCE_COLORS = frozenset(REFERENCE_COLOR_COUNTS)


def extract_reference_icons() -> dict[str, Image.Image]:
    """Extract the four gold resource icons directly from the paper figure."""
    with Image.open(REFERENCE_FIGURE) as source_image:
        source = source_image.convert("RGB")
    if source.size != (790, 345):
        raise RuntimeError(
            f"Unexpected reference-figure size {source.size}; "
            "the icon crop coordinates need to be reviewed"
        )

    icons: dict[str, Image.Image] = {}
    for name, box in ICON_SOURCE_WINDOWS.items():
        crop = np.asarray(source.crop(box), dtype=np.uint8)
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(
            np.int16
        )
        gold_mask = (
            (chroma >= 2)
            & (red > 120)
            & (red >= green)
            & (green >= blue)
        )
        ys, xs = np.nonzero(gold_mask)
        if len(xs) == 0:
            raise RuntimeError(f"No gold pixels found for the {name} reference icon")
        rgba = np.zeros((*crop.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = crop
        rgba[:, :, 3] = gold_mask.astype(np.uint8) * 255
        tight = rgba[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        icons[name] = Image.fromarray(tight)
    return icons


RESOURCE_ICONS = extract_reference_icons()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge0 == edge1:
        return float(value >= edge1)
    x = clamp((value - edge0) / (edge1 - edge0))
    return x * x * (3.0 - 2.0 * x)


def hex_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def blend(color: str | tuple[int, int, int], opacity: float) -> tuple[int, int, int]:
    foreground = hex_rgb(color) if isinstance(color, str) else color
    background = hex_rgb(BACKGROUND)
    opacity = clamp(opacity)
    return tuple(
        round(background[index] * (1.0 - opacity) + foreground[index] * opacity)
        for index in range(3)
    )


def scaled_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(round(value * SCALE) for value in box)  # type: ignore[return-value]


def scaled_points(points: Iterable[tuple[float, float]]) -> list[tuple[int, int]]:
    return [(round(x * SCALE), round(y * SCALE)) for x, y in points]


def font_path(bold: bool = False) -> str | None:
    filename = "lmroman10-bold.otf" if bold else "lmroman10-regular.otf"
    latin_modern_candidates = [
        Path.home() / "Library" / "Fonts" / filename,
        Path("/Library/Fonts") / filename,
        Path("/usr/share/fonts/opentype/lmodern") / filename,
    ]
    texlive_root = Path("/usr/local/texlive")
    if texlive_root.exists():
        latin_modern_candidates.extend(
            texlive_root.glob(
                f"*/texmf-dist/fonts/opentype/public/lm/{filename}"
            )
        )
    exact_font = next(
        (str(candidate) for candidate in latin_modern_candidates if candidate.exists()),
        None,
    )

    matplotlib_font = None
    if MATPLOTLIB_FONT_DIR is not None:
        fallback_filename = "STIXGeneralBol.ttf" if bold else "STIXGeneral.ttf"
        candidate = MATPLOTLIB_FONT_DIR / fallback_filename
        if candidate.exists():
            matplotlib_font = str(candidate)
    candidates = (
        [path for path in [
            exact_font,
            matplotlib_font,
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        ] if path]
        if bold
        else [path for path in [
            exact_font,
            matplotlib_font,
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ] if path]
    )
    return next((candidate for candidate in candidates if Path(candidate).exists()), None)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = font_path(bold)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(path, size * SCALE)


FONTS = {
    "title": load_font(23, bold=True),
    "section": load_font(11, bold=True),
    "body": load_font(13),
    "body_bold": load_font(13, bold=True),
    "small": load_font(11),
    "small_bold": load_font(11, bold=True),
    "equation": load_font(14),
    "footer": load_font(16, bold=True),
}


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    value: str,
    *,
    font: str = "body",
    fill: str | tuple[int, int, int] = INK,
    opacity: float = 1.0,
    anchor: str = "la",
) -> None:
    if opacity <= 0.002:
        return
    color = blend(fill, opacity) if isinstance(fill, str) else fill
    draw.text(
        (round(xy[0] * SCALE), round(xy[1] * SCALE)),
        value,
        font=FONTS[font],
        fill=color,
        anchor=anchor,
    )


def dashed_line(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    *,
    fill: str,
    opacity: float,
    width: int = 2,
    dash: int = 7,
    gap: int = 5,
) -> None:
    if len(points) < 2:
        return
    distance_so_far = 0.0
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        distance = math.hypot(x1 - x0, y1 - y0)
        if distance == 0:
            continue
        ux = (x1 - x0) / distance
        uy = (y1 - y0) / distance
        cursor = 0.0
        while cursor < distance:
            pattern_position = (distance_so_far + cursor) % (dash + gap)
            if pattern_position < dash:
                step = min(dash - pattern_position, distance - cursor)
                draw.line(
                    scaled_points(
                        [
                            (x0 + ux * cursor, y0 + uy * cursor),
                            (x0 + ux * (cursor + step), y0 + uy * (cursor + step)),
                        ]
                    ),
                    fill=blend(fill, opacity),
                    width=width * SCALE,
                )
            else:
                step = min(dash + gap - pattern_position, distance - cursor)
            cursor += step
        distance_so_far += distance


def draw_resource_chip(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    label: str,
    icon: str,
    opacity: float,
) -> None:
    if opacity <= 0.002:
        return
    x0, y0, _x1, y1 = box
    icon_center = (x0 + 16, (y0 + y1) / 2)
    draw.ellipse(
        scaled_box(
            (
                icon_center[0] - 15,
                icon_center[1] - 15,
                icon_center[0] + 15,
                icon_center[1] + 15,
            )
        ),
        fill=blend(PANEL if icon == "athermality" else GOLD_PALE, opacity),
    )

    source_icon = RESOURCE_ICONS[icon]
    target_extent = 25 * SCALE
    icon_scale = min(
        target_extent / source_icon.width,
        target_extent / source_icon.height,
    )
    icon_image = source_icon.resize(
        (
            max(1, round(source_icon.width * icon_scale)),
            max(1, round(source_icon.height * icon_scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    if opacity < 0.999:
        alpha = icon_image.getchannel("A").point(lambda value: round(value * opacity))
        icon_image.putalpha(alpha)
    paste_position = (
        round(icon_center[0] * SCALE - icon_image.width / 2),
        round(icon_center[1] * SCALE - icon_image.height / 2),
    )
    image.paste(icon_image, paste_position, icon_image)
    text(
        draw,
        (x0 + 36, (y0 + y1) / 2),
        label,
        font="body",
        fill=INK,
        opacity=opacity,
        anchor="lm",
    )


def monotone_a(time: np.ndarray | float) -> np.ndarray | float:
    return A_SLOW_WEIGHT * np.exp(-SLOW_DECAY * time) + A_FAST_WEIGHT * np.exp(
        -FAST_DECAY * time
    )


def monotone_b(time: np.ndarray | float) -> np.ndarray | float:
    return B_SLOW_WEIGHT * np.exp(-SLOW_DECAY * time) + B_FAST_WEIGHT * np.exp(
        -FAST_DECAY * time
    )


def draw_state_card(
    draw: ImageDraw.ImageDraw,
    *,
    box: tuple[float, float, float, float],
    state: str,
    description: str,
    value: float,
    slow_weight: float,
    slow_label: str,
    color: str,
    opacity: float,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(
        scaled_box(box),
        radius=11 * SCALE,
        fill=blend(PANEL, opacity * 0.86),
        outline=blend(STRUCTURE, opacity),
        width=1 * SCALE,
    )
    draw.ellipse(
        scaled_box((x0 + 12, y0 + 12, x0 + 22, y0 + 22)),
        fill=blend(color, opacity),
    )
    text(
        draw,
        (x0 + 29, y0 + 17),
        state,
        font="body_bold",
        opacity=opacity,
        anchor="lm",
    )
    text(
        draw,
        (x0 + 58, y0 + 17),
        description,
        font="small",
        fill=MUTED,
        opacity=opacity,
        anchor="lm",
    )
    text(
        draw,
        (x0 + 12, y0 + 39),
        "chosen M",
        font="small",
        fill=MUTED,
        opacity=opacity,
        anchor="lm",
    )
    bar_x0, bar_x1 = x0 + 72, x1 - 12
    draw.rounded_rectangle(
        scaled_box((bar_x0, y0 + 35, bar_x1, y0 + 43)),
        radius=4 * SCALE,
        fill=blend(STRUCTURE, opacity * 0.55),
    )
    draw.rounded_rectangle(
        scaled_box((bar_x0, y0 + 35, bar_x0 + (bar_x1 - bar_x0) * value, y0 + 43)),
        radius=4 * SCALE,
        fill=blend(color, opacity),
    )
    text(
        draw,
        (x0 + 12, y0 + 59),
        "slow weight",
        font="small",
        fill=MUTED,
        opacity=opacity,
        anchor="lm",
    )
    slow_bar_x0, slow_bar_x1 = x0 + 86, x1 - 52
    draw.rounded_rectangle(
        scaled_box((slow_bar_x0, y0 + 55, slow_bar_x1, y0 + 63)),
        radius=4 * SCALE,
        fill=blend(STRUCTURE, opacity * 0.55),
    )
    draw.rounded_rectangle(
        scaled_box(
            (
                slow_bar_x0,
                y0 + 55,
                slow_bar_x0 + (slow_bar_x1 - slow_bar_x0) * slow_weight,
                y0 + 63,
            )
        ),
        radius=4 * SCALE,
        fill=blend(SLOW_MODE_COLOR, opacity),
    )
    text(
        draw,
        (x1 - 11, y0 + 59),
        slow_label,
        font="small_bold",
        fill=MUTED,
        opacity=opacity,
        anchor="rm",
    )


def draw_free_set(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    opacity: float,
    glow: float = 0.0,
) -> None:
    if opacity <= 0.002:
        return
    cx, cy = center
    if glow > 0.001:
        glow_extra = 5.0 + 4.0 * glow
        draw.ellipse(
            scaled_box(
                (
                    cx - 34 - glow_extra,
                    cy - 67 - glow_extra,
                    cx + 34 + glow_extra,
                    cy + 67 + glow_extra,
                )
            ),
            outline=blend(SLOW_MODE_COLOR, opacity * glow * 0.55),
            width=2 * SCALE,
        )
    draw.ellipse(
        scaled_box((cx - 34, cy - 67, cx + 34, cy + 67)),
        fill=blend(FREE_SET_COLOR, opacity * (0.11 + 0.08 * glow)),
        outline=blend(FREE_SET_COLOR, opacity * (0.75 + 0.2 * glow)),
        width=2 * SCALE,
    )
    text(
        draw,
        (cx, cy - 52),
        "Free-state",
        font="small_bold",
        fill=FREE_SET_COLOR,
        opacity=opacity,
        anchor="mm",
    )
    text(
        draw,
        (cx, cy - 36),
        "set F",
        font="body_bold",
        fill=FREE_SET_COLOR,
        opacity=opacity,
        anchor="mm",
    )


def draw_token(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    color: str,
    slow_strength: float,
    opacity: float,
    label: str,
) -> None:
    cx, cy = center
    glow_radius = radius + 5 + 6 * slow_strength
    draw.ellipse(
        scaled_box(
            (
                cx - glow_radius,
                cy - glow_radius,
                cx + glow_radius,
                cy + glow_radius,
            )
        ),
        fill=blend(SLOW_MODE_COLOR, opacity * 0.10 * (0.25 + slow_strength)),
        outline=blend(SLOW_MODE_COLOR, opacity * 0.45 * slow_strength),
        width=1 * SCALE,
    )
    draw.ellipse(
        scaled_box((cx - radius, cy - radius, cx + radius, cy + radius)),
        fill=blend(color, opacity * 0.84),
        outline=blend(color, opacity),
        width=2 * SCALE,
    )
    text(
        draw,
        (cx, cy),
        label,
        font="small_bold",
        fill=WHITE,
        opacity=opacity,
        anchor="mm",
    )


def plot_point(
    plot_box: tuple[float, float, float, float], time_value: float, monotone: float
) -> tuple[float, float]:
    x0, y0, x1, y1 = plot_box
    x = x0 + (x1 - x0) * time_value / 8.0
    y = y1 - (y1 - y0) * monotone / 1.05
    return x, y


def draw_plot(
    draw: ImageDraw.ImageDraw,
    progress: float,
    opacity: float,
    crossing_pulse: float,
) -> None:
    plot = (668.0, 135.0, 930.0, 307.0)
    x0, y0, x1, y1 = plot
    text(
        draw,
        (x0, y0 - 23),
        "Chosen resource monotone M",
        font="section",
        opacity=opacity,
        anchor="la",
    )
    draw.line(
        scaled_points([(x0, y0), (x0, y1), (x1, y1)]),
        fill=blend(INK, opacity),
        width=2 * SCALE,
    )
    for fraction in (0.25, 0.5, 0.75):
        grid_y = y1 - (y1 - y0) * fraction / 1.05
        draw.line(
            scaled_points([(x0, grid_y), (x1, grid_y)]),
            fill=blend(GRID, opacity),
            width=1 * SCALE,
        )
    text(draw, (x0 - 5, y1 + 2), "0", font="small", opacity=opacity, anchor="ra")
    text(draw, (x1, y1 + 4), "time", font="small", opacity=opacity, anchor="ra")

    times = np.linspace(0.0, 8.0, 180)
    values_a = monotone_a(times)
    values_b = monotone_b(times)
    all_a = [plot_point(plot, float(t), float(v)) for t, v in zip(times, values_a)]
    all_b = [plot_point(plot, float(t), float(v)) for t, v in zip(times, values_b)]
    draw.line(
        scaled_points(all_a),
        fill=blend(STATE_A_COLOR, opacity * 0.16),
        width=2 * SCALE,
    )
    dashed_line(
        draw,
        all_b,
        fill=STATE_B_COLOR,
        opacity=opacity * 0.16,
        width=2,
        dash=6,
        gap=5,
    )

    last_index = max(1, round(progress * (len(times) - 1)))
    active_a = all_a[: last_index + 1]
    active_b = all_b[: last_index + 1]
    draw.line(
        scaled_points(active_a),
        fill=blend(STATE_A_COLOR, opacity),
        width=3 * SCALE,
        joint="curve",
    )
    dashed_line(
        draw,
        active_b,
        fill=STATE_B_COLOR,
        opacity=opacity,
        width=3,
        dash=7,
        gap=5,
    )

    legend_y = y0 + 10
    draw.line(
        scaled_points([(x0 + 13, legend_y), (x0 + 37, legend_y)]),
        fill=blend(STATE_A_COLOR, opacity),
        width=3 * SCALE,
    )
    text(
        draw,
        (x0 + 43, legend_y),
        "A",
        font="small_bold",
        opacity=opacity,
        anchor="lm",
    )
    dashed_line(
        draw,
        [(x0 + 86, legend_y), (x0 + 110, legend_y)],
        fill=STATE_B_COLOR,
        opacity=opacity,
        width=3,
        dash=6,
        gap=4,
    )
    text(
        draw,
        (x0 + 116, legend_y),
        "B",
        font="small_bold",
        opacity=opacity,
        anchor="lm",
    )

    crossing = plot_point(plot, CROSSING_TIME, CROSSING_VALUE)
    if progress >= CROSSING_TIME / 8.0:
        marker_opacity = opacity * smoothstep(
            CROSSING_TIME / 8.0, CROSSING_TIME / 8.0 + 0.07, progress
        )
        draw.line(
            scaled_points([(crossing[0], crossing[1]), (crossing[0], y1)]),
            fill=blend(GOLD, marker_opacity * 0.8),
            width=1 * SCALE,
        )
        radius = 5.0 + 3.0 * crossing_pulse
        draw.ellipse(
            scaled_box(
                (
                    crossing[0] - radius,
                    crossing[1] - radius,
                    crossing[0] + radius,
                    crossing[1] + radius,
                )
            ),
            fill=blend(GOLD, marker_opacity * 0.25),
            outline=blend(GOLD, marker_opacity),
            width=2 * SCALE,
        )
        text(
            draw,
            (crossing[0] + 16, crossing[1] - 10),
            "Mpemba crossing",
            font="small_bold",
            fill=INK,
            opacity=marker_opacity,
            anchor="lm",
        )
        text(
            draw,
            (crossing[0], y1 + 10),
            "crossing time",
            font="small_bold",
            fill=GOLD_DARK,
            opacity=marker_opacity,
            anchor="ma",
        )

    text(
        draw,
        ((x0 + x1) / 2, y1 + 25),
        "schematic Markovian example",
        font="small",
        fill=MUTED,
        opacity=opacity * 0.8,
        anchor="ma",
    )


def render_frame(frame_index: int) -> Image.Image:
    time_seconds = frame_index / FPS
    image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), hex_rgb(BACKGROUND))
    draw = ImageDraw.Draw(image)

    chrome_opacity = 1.0
    content_opacity = smoothstep(0.24, 0.62, time_seconds) * (
        1.0 - smoothstep(7.12, 7.60, time_seconds)
    )

    text(
        draw,
        (WIDTH / 2, 18),
        "Different resources - one Mpemba criterion",
        font="title",
        opacity=chrome_opacity,
        anchor="mm",
    )

    chip_boxes = [
        (88.0, 47.0, 245.0, 79.0),
        (258.0, 47.0, 405.0, 79.0),
        (418.0, 47.0, 597.0, 79.0),
        (610.0, 47.0, 865.0, 79.0),
    ]
    chip_specs = [
        ("Athermality", "athermality"),
        ("Asymmetry", "asymmetry"),
        ("Nonstationarity", "nonstationarity"),
        ("Quantum coherence", "coherence"),
    ]
    for box, (label, icon) in zip(chip_boxes, chip_specs):
        draw_resource_chip(
            image,
            draw,
            box,
            label,
            icon,
            chrome_opacity,
        )

    section_opacity = content_opacity
    text(
        draw,
        (28, 100),
        "Initial states",
        font="section",
        opacity=section_opacity,
    )
    text(
        draw,
        (289, 100),
        "The same free dynamics for both states",
        font="section",
        opacity=section_opacity,
    )

    card_a_opacity = section_opacity
    card_b_opacity = section_opacity
    draw_state_card(
        draw,
        box=(28.0, 118.0, 262.0, 188.0),
        state="A",
        description="initially more resourceful",
        value=float(monotone_a(0.0)),
        slow_weight=A_SLOW_WEIGHT / B_SLOW_WEIGHT,
        slow_label="small",
        color=STATE_A_COLOR,
        opacity=card_a_opacity,
    )
    draw_state_card(
        draw,
        box=(28.0, 202.0, 262.0, 272.0),
        state="B",
        description="initially less resourceful",
        value=float(monotone_b(0.0)),
        slow_weight=1.0,
        slow_label="large",
        color=STATE_B_COLOR,
        opacity=card_b_opacity,
    )
    text(
        draw,
        (145, 290),
        "slowest mode relevant to the chosen resource",
        font="small",
        fill=MUTED,
        opacity=section_opacity,
        anchor="ma",
    )

    machine_box = (286.0, 118.0, 638.0, 307.0)
    draw.rounded_rectangle(
        scaled_box(machine_box),
        radius=16 * SCALE,
        fill=blend(PANEL_ALT, section_opacity * 0.78),
        outline=blend(STRUCTURE, section_opacity),
        width=2 * SCALE,
    )
    text(
        draw,
        (305, 139),
        "E(t)",
        font="equation",
        fill=INK,
        opacity=section_opacity,
        anchor="lm",
    )
    text(
        draw,
        (337, 139),
        "shared decay spectrum",
        font="small",
        fill=MUTED,
        opacity=section_opacity,
        anchor="lm",
    )

    track_x0, track_x1 = 350.0, 565.0
    row_a, row_b = 184.0, 250.0
    for row_y in (row_a, row_b):
        draw.line(
            scaled_points([(track_x0, row_y), (track_x1, row_y)]),
            fill=blend(STRUCTURE, section_opacity),
            width=2 * SCALE,
        )
        draw.polygon(
            scaled_points(
                [
                    (track_x1, row_y),
                    (track_x1 - 7, row_y - 4),
                    (track_x1 - 7, row_y + 4),
                ]
            ),
            fill=blend(STRUCTURE, section_opacity),
        )
    text(
        draw,
        (306, row_a),
        "A",
        font="body_bold",
        fill=STATE_A_COLOR,
        opacity=section_opacity,
        anchor="mm",
    )
    text(
        draw,
        (306, row_b),
        "B",
        font="body_bold",
        fill=STATE_B_COLOR,
        opacity=section_opacity,
        anchor="mm",
    )

    evolution_progress = smoothstep(1.25, 5.75, time_seconds)
    physical_time = 8.0 * evolution_progress
    value_a = float(monotone_a(physical_time))
    value_b = float(monotone_b(physical_time))
    slow_a = math.sqrt(A_SLOW_WEIGHT) * math.exp(
        -(SLOW_DECAY / 2.0) * physical_time
    )
    slow_b = math.sqrt(B_SLOW_WEIGHT) * math.exp(
        -(SLOW_DECAY / 2.0) * physical_time
    )

    # Tokens follow the arrow tracks and then curve into the free-state set F,
    # illustrating that the free dynamics drives every state toward F.
    free_center_x, free_center_y = 600.0, 220.0
    token_target_a = (free_center_x, free_center_y - 20.0)
    token_target_b = (free_center_x, free_center_y + 30.0)
    along_track_p = clamp(evolution_progress / 0.72)
    capture_progress = smoothstep(0.72, 1.0, evolution_progress)
    track_position_x = track_x0 + (track_x1 - track_x0) * along_track_p
    token_x_a = (
        track_position_x
        + (token_target_a[0] - track_position_x) * capture_progress
    )
    token_x_b = (
        track_position_x
        + (token_target_b[0] - track_position_x) * capture_progress
    )
    token_y_a = row_a + (token_target_a[1] - row_a) * capture_progress
    token_y_b = row_b + (token_target_b[1] - row_b) * capture_progress
    capture_alpha = 1.0 - 0.35 * capture_progress
    token_opacity = section_opacity

    # Draw the free-state set first so the tokens appear to move into it.
    free_opacity = section_opacity
    draw_free_set(
        draw,
        (free_center_x, free_center_y),
        free_opacity,
        glow=capture_progress,
    )

    draw_token(
        draw,
        center=(token_x_a, token_y_a),
        radius=7.0 + 16.0 * value_a**0.45,
        color=STATE_A_COLOR,
        slow_strength=slow_a,
        opacity=token_opacity * capture_alpha,
        label="A",
    )
    draw_token(
        draw,
        center=(token_x_b, token_y_b),
        radius=7.0 + 16.0 * value_b**0.45,
        color=STATE_B_COLOR,
        slow_strength=slow_b,
        opacity=token_opacity * capture_alpha,
        label="B",
    )

    # The pulse is aesthetic; the exact crossing is placed from the analytic curve.
    crossing_pulse = math.exp(-((time_seconds - 3.0) / 0.28) ** 2)
    draw_plot(draw, evolution_progress, section_opacity, crossing_pulse)

    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def make_global_palette(_frames: Sequence[Image.Image]) -> Image.Image:
    """Build a GIF palette exclusively from colors present in Figure 1."""
    design_colors = [hex_rgb(color) for color in DESIGN_COLOR_HEXES]
    palette_colors: list[tuple[int, int, int]] = []
    reference_colors_by_frequency = [
        color for color, _count in REFERENCE_COLOR_COUNTS.most_common()
    ]
    for color in design_colors + reference_colors_by_frequency:
        if color not in palette_colors:
            palette_colors.append(color)
        if len(palette_colors) == 256:
            break
    palette_colors = palette_colors[:256]
    palette_colors.extend([(0, 0, 0)] * (256 - len(palette_colors)))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette(
        [component for color in palette_colors for component in color]
    )
    return palette_image


def quantize_with_palette(frame: Image.Image, palette: Image.Image) -> Image.Image:
    """Quantize while keeping every exact Figure-1 design token unchanged."""
    quantized = frame.quantize(palette=palette, dither=Image.Dither.NONE)
    indices = np.asarray(quantized, dtype=np.uint8).copy()
    source = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    for palette_index, color in enumerate(
        hex_rgb(value) for value in DESIGN_COLOR_HEXES
    ):
        exact_pixels = np.all(source == color, axis=2)
        indices[exact_pixels] = palette_index
    result = Image.new("P", frame.size)
    result.putdata(indices.reshape(-1))
    result.putpalette(palette.getpalette())
    return result


def save_gif(frames: Sequence[Image.Image], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    palette = make_global_palette(frames)
    paletted_frames = [quantize_with_palette(frame, palette) for frame in frames]
    # GIF stores durations in centiseconds.  Distribute the rounding error
    # across frames so the complete loop remains exactly DURATION seconds.
    total_centiseconds = round(DURATION * 100)
    base_duration = total_centiseconds // len(frames)
    remainder = total_centiseconds % len(frames)
    accumulator = 0
    durations_ms: list[int] = []
    for _ in frames:
        accumulator += remainder
        extra = 0
        if accumulator >= len(frames):
            extra = 1
            accumulator -= len(frames)
        durations_ms.append((base_duration + extra) * 10)
    paletted_frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=paletted_frames[1:],
        duration=durations_ms,
        loop=0,
        optimize=True,
        disposal=1,
    )


def validate_model() -> None:
    if not math.isclose(float(monotone_a(0.0)), 1.0):
        raise RuntimeError("M_A(0) is no longer normalized to one")
    if not math.isclose(float(monotone_b(0.0)), 0.75):
        raise RuntimeError("M_B(0) is no longer equal to three quarters")
    if not math.isclose(float(monotone_a(CROSSING_TIME)), CROSSING_VALUE):
        raise RuntimeError("The analytic crossing value is inconsistent with M_A")
    if not math.isclose(float(monotone_b(CROSSING_TIME)), CROSSING_VALUE):
        raise RuntimeError("The analytic crossing value is inconsistent with M_B")
    sample_times = np.linspace(0.0, 8.0, 1000)
    values_a = np.asarray(monotone_a(sample_times))
    values_b = np.asarray(monotone_b(sample_times))
    if not (np.all(np.diff(values_a) < 0.0) and np.all(np.diff(values_b) < 0.0)):
        raise RuntimeError("Both schematic monotones must decrease strictly")
    before = sample_times < CROSSING_TIME
    after = sample_times > CROSSING_TIME
    if not (np.all(values_a[before] > values_b[before]) and np.all(values_a[after] < values_b[after])):
        raise RuntimeError("The schematic curves must cross exactly once")


def validate_gif(output: Path, expected_frames: int) -> None:
    with Image.open(output) as animation:
        if animation.format != "GIF":
            raise RuntimeError(f"Unexpected image format: {animation.format}")
        if animation.size != (WIDTH, HEIGHT):
            raise RuntimeError(f"Unexpected GIF size: {animation.size}")
        # Pillow merges adjacent identical frames during optimization.
        if animation.n_frames < expected_frames * 0.6:
            raise RuntimeError(
                f"Too many frames were lost during optimization: "
                f"{animation.n_frames} of {expected_frames}"
            )
        if animation.info.get("loop") != 0:
            raise RuntimeError("GIF is not configured to loop continuously")
        durations = []
        unexpected_colors: set[tuple[int, int, int]] = set()
        for frame_index in range(animation.n_frames):
            animation.seek(frame_index)
            durations.append(animation.info.get("duration", 0))
            colors = animation.convert("RGB").getcolors(
                maxcolors=animation.width * animation.height
            )
            if colors is None:
                raise RuntimeError("Could not enumerate GIF colors")
            unexpected_colors.update(
                color for _count, color in colors if color not in REFERENCE_COLORS
            )
        if unexpected_colors:
            sample = sorted(unexpected_colors)[:8]
            raise RuntimeError(
                "GIF contains colors outside the Figure-1 neutral/gold palette: "
                f"{sample}"
            )
        expected_duration = DURATION * 1000
        if abs(sum(durations) - expected_duration) > 20:
            raise RuntimeError(
                f"Unexpected duration: {sum(durations)} ms "
                f"(expected about {expected_duration:.0f} ms)"
            )
        animation.seek(0)
        first_frame = animation.convert("RGB")
        animation.seek(animation.n_frames - 1)
        last_frame = animation.convert("RGB")
        if ImageChops.difference(first_frame, last_frame).getbbox() is not None:
            raise RuntimeError("First and last frames differ; the loop is not seamless")
        blank = Image.new("RGB", animation.size, hex_rgb(BACKGROUND))
        if ImageChops.difference(first_frame, blank).getbbox() is None:
            raise RuntimeError("The static preview frame is blank")
    if output.stat().st_size > 5 * 1024 * 1024:
        raise RuntimeError("GIF exceeds the 5 MiB README size budget")


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "figures" / "resource_mpemba_mechanism_fig1.gif",
        help="Destination GIF path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help=f"Frames per second (default: {FPS}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 10 <= args.fps <= 30:
        raise ValueError("--fps must be between 10 and 30")
    frame_count = round(DURATION * args.fps)
    global FPS
    FPS = args.fps
    validate_model()
    frames = [render_frame(index) for index in range(frame_count)]
    temporary_output = args.output.with_name(
        f".{args.output.stem}.tmp{args.output.suffix}"
    )
    try:
        save_gif(frames, temporary_output)
        validate_gif(temporary_output, frame_count)
        temporary_output.replace(args.output)
    finally:
        temporary_output.unlink(missing_ok=True)
    size_mib = args.output.stat().st_size / (1024 * 1024)
    print(
        f"Wrote {args.output} ({WIDTH}x{HEIGHT}, {frame_count} frames, "
        f"{size_mib:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
