"""
Object selection verification game.

Click one of the colored boxes; a result screen shows whether you "verified" or not.
No pynput required — uses only matplotlib for input and visuals.
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_OBJECTS = 3
RECT_SIZE = 0.12
BOX_COLORS = ["#4ECDC4", "#FF6B6B", "#95E1D3"]  # teal, coral, mint
BOX_EDGE = "#2D3436"
HOVER_ALPHA = 0.95
IDLE_ALPHA = 0.75
BG_COLOR = "#F8F9FA"
SUCCESS_COLOR = "#00B894"
FAIL_COLOR = "#D63031"


# ---------------------------------------------------------------------------
# Game logic
# ---------------------------------------------------------------------------


def generate_frame(n: int = NUM_OBJECTS) -> list[tuple[float, float]]:
    """Random (x, y) in [0.15, 0.85] so boxes stay on screen."""
    return [
        (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
        for _ in range(n)
    ]


def point_in_rect(px: float, py: float, cx: float, cy: float, half: float) -> bool:
    """True if (px, py) is inside the axis-aligned rect centered at (cx, cy) with half-side half."""
    return (
        (px is not None and py is not None)
        and (cx - half <= px <= cx + half)
        and (cy - half <= py <= cy + half)
    )


def hit_test(click_x: float, click_y: float, coords: list, half: float) -> int | None:
    """Return index of the box containing (click_x, click_y), or None."""
    for i, (cx, cy) in enumerate(coords):
        if point_in_rect(click_x, click_y, cx, cy, half):
            return i
    return None


# ---------------------------------------------------------------------------
# Main game
# ---------------------------------------------------------------------------


def run_game():
    coords = generate_frame(NUM_OBJECTS)
    half = RECT_SIZE / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Click one of the boxes to verify", fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Draw boxes
    rects: list[mpatches.Rectangle] = []
    for i, (cx, cy) in enumerate(coords):
        rect = mpatches.Rectangle(
            (cx - half, cy - half),
            RECT_SIZE,
            RECT_SIZE,
            facecolor=BOX_COLORS[i % len(BOX_COLORS)],
            edgecolor=BOX_EDGE,
            linewidth=2,
            alpha=IDLE_ALPHA,
            picker=5,
        )
        ax.add_patch(rect)
        rects.append(rect)

    result = {"selected_index": None, "done": False}

    def on_hover(event):
        if result["done"] or event.inaxes != ax:
            return
        for i, r in enumerate(rects):
            cx, cy = coords[i][0], coords[i][1]
            if point_in_rect(event.xdata, event.ydata, cx, cy, half):
                r.set_alpha(HOVER_ALPHA)
                fig.canvas.draw_idle()
                return
        for r in rects:
            r.set_alpha(IDLE_ALPHA)
        fig.canvas.draw_idle()

    def on_click(event):
        if result["done"] or event.inaxes != ax or event.button != 1:
            return
        idx = hit_test(event.xdata, event.ydata, coords, half)
        result["selected_index"] = idx
        result["done"] = True
        plt.close(fig)

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    return result["selected_index"] is not None


# ---------------------------------------------------------------------------
# Result screen (visual outcome after run)
# ---------------------------------------------------------------------------


def show_result(verified: bool):
    """Show a full-screen result figure with clear visual feedback."""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(SUCCESS_COLOR if verified else FAIL_COLOR)
    ax.set_facecolor(SUCCESS_COLOR if verified else FAIL_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if verified:
        title = "You're verified!"
        emoji = "✓"
        sub = "Nice click."
    else:
        title = "Try again"
        emoji = "✗"
        sub = "Click inside one of the boxes next time."

    ax.text(0.5, 0.6, emoji, fontsize=80, ha="center", va="center", color="white")
    ax.text(0.5, 0.35, title, fontsize=28, ha="center", va="center", color="white", weight="bold")
    ax.text(0.5, 0.2, sub, fontsize=14, ha="center", va="center", color="white", alpha=0.9)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    verified = run_game()
    show_result(verified)
