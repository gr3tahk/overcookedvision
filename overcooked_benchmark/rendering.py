from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CELL_SIZE = 96
PADDING = 18
LEGEND_HEIGHT = 104

TILE_COLORS = {
    " ": "#f4ead7",
    "X": "#7c756a",
    "O": "#d98632",
    "D": "#77a8d9",
    "P": "#c85a54",
    "S": "#7fbe7a",
}
TILE_LABELS = {
    " ": "floor",
    "X": "counter",
    "O": "onion",
    "D": "dish",
    "P": "pot",
    "S": "serve",
}
PLAYER_COLORS = ["#ffd85c", "#72d6c9"]


def _font(size: int):
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _text_center(draw: ImageDraw.ImageDraw, xyxy, text: str, font, fill="#1f2933"):
    left, top, right, bottom = xyxy
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((left + (right - left - width) / 2, top + (bottom - top - height) / 2), text, font=font, fill=fill)


def _held_name(player: dict[str, Any]) -> str:
    held = player.get("held_object")
    return held.get("name", "nothing") if held else "nothing"


def _pot_label(state, pos) -> str:
    if not state.has_object(pos):
        return "P 0/3"
    soup = state.get_object(pos)
    count = len(soup.ingredients)
    if soup.is_ready:
        stage = "ready"
    elif soup.is_cooking:
        stage = "cook"
    else:
        stage = "fill"
    return f"P {count}/3 {stage}"


def render_state_image(state, mdp, tick: int = 0, score: int = 0) -> Image.Image:
    width = mdp.width * CELL_SIZE + PADDING * 2
    height = mdp.height * CELL_SIZE + PADDING * 2 + LEGEND_HEIGHT
    image = Image.new("RGB", (width, height), "#fbf7ef")
    draw = ImageDraw.Draw(image)
    title_font = _font(22)
    label_font = _font(15)
    small_font = _font(12)

    draw.text((PADDING, 8), f"Overcooked tick {tick} | score {score}", font=title_font, fill="#1f2933")
    board_top = PADDING + 28

    for y in range(mdp.height):
        for x in range(mdp.width):
            tile = mdp.get_terrain_type_at_pos((x, y))
            left = PADDING + x * CELL_SIZE
            top = board_top + y * CELL_SIZE
            right = left + CELL_SIZE
            bottom = top + CELL_SIZE
            draw.rounded_rectangle(
                (left + 2, top + 2, right - 2, bottom - 2),
                radius=8,
                fill=TILE_COLORS.get(tile, "#d6d0c4"),
                outline="#312f2b",
                width=2,
            )
            tile_label = TILE_LABELS.get(tile, tile)
            if tile == "P":
                tile_label = _pot_label(state, (x, y))
            _text_center(draw, (left + 5, top + 5, right - 5, bottom - 5), tile_label, label_font, fill="#111827")

    for obj in state.to_dict()["objects"]:
        x, y = obj["position"]
        left = PADDING + x * CELL_SIZE + 12
        top = board_top + y * CELL_SIZE + 58
        draw.rounded_rectangle((left, top, left + 72, top + 26), radius=6, fill="#fdfdfd", outline="#111827")
        _text_center(draw, (left, top, left + 72, top + 26), obj["name"], small_font, fill="#111827")

    state_dict = state.to_dict()
    for player_id, player in enumerate(state_dict["players"]):
        x, y = player["position"]
        left = PADDING + x * CELL_SIZE + 15
        top = board_top + y * CELL_SIZE + 15
        draw.ellipse((left, top, left + 66, top + 66), fill=PLAYER_COLORS[player_id], outline="#111827", width=3)
        _text_center(draw, (left, top + 3, left + 66, top + 33), f"P{player_id}", title_font, fill="#111827")
        held = _held_name(player)
        _text_center(draw, (left - 10, top + 35, left + 76, top + 62), held, small_font, fill="#111827")
        orientation = tuple(player["orientation"])
        direction = {(0, -1): "N", (0, 1): "S", (1, 0): "E", (-1, 0): "W", (0, 0): "-"}[orientation]
        draw.text((left + 50, top + 2), direction, font=small_font, fill="#111827")

    legend_top = board_top + mdp.height * CELL_SIZE + 12
    legend = "Legend: P0=Alice, P1=Bob. Tiles: onion dispenser, dish dispenser, pot, serve, counter, floor."
    controls = "Actions: up, down, left, right move/turn; interact uses the tile you face; stay waits."
    draw.text((PADDING, legend_top), legend, font=label_font, fill="#1f2933")
    draw.text((PADDING, legend_top + 28), controls, font=label_font, fill="#1f2933")
    draw.text((PADDING, legend_top + 56), "Goal: cook 3 onions in a pot, get dish, pick up soup, deliver.", font=label_font, fill="#1f2933")
    return image


def render_state_data_url(state, mdp, tick: int = 0, score: int = 0) -> str:
    image = render_state_image(state, mdp, tick=tick, score=score)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
