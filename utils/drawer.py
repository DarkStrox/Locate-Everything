from PIL import Image, ImageDraw, ImageFont, ImageColor
from typing import List, Tuple
import colorsys
from utils.parser import DetectionBox

# Vibrant color palette for bounding box annotations
COLOR_PALETTE = [
    "#FF3838", "#FF9D00", "#FFB000", "#15CD72", "#00C2FF",
    "#3A86FF", "#8338EC", "#FF007F", "#00F5D4", "#7B2CBF",
    "#00E676", "#FFEA00", "#FF4081", "#7C4DFF", "#00E5FF"
]

def get_color_for_label(label: str, label_colors: dict) -> str:
    if label not in label_colors:
        idx = len(label_colors) % len(COLOR_PALETTE)
        label_colors[label] = COLOR_PALETTE[idx]
    return label_colors[label]

def hex_to_rgba(hex_color: str, alpha: int = 40) -> Tuple[int, int, int, int]:
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)

def draw_annotations(
    image: Image.Image,
    detections: List[DetectionBox],
    box_thickness: int = 3,
    font_size: int = 14,
    show_labels: bool = True,
    fill_alpha: int = 45
) -> Image.Image:
    """
    Draws bounding boxes and label badges onto a PIL image.
    """
    if image is None:
        return None

    # Work on a copy
    annotated = image.copy().convert("RGBA")
    width, height = annotated.size

    # Create overlay for semi-transparent fills
    overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    annotated_draw = ImageDraw.Draw(annotated)

    label_colors = {}

    # Try loading default font or PIL basic font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    for idx, det in enumerate(detections, start=1):
        xmin, ymin, xmax, ymax = det.to_pixel_box(width, height)
        color_hex = get_color_for_label(det.label, label_colors)
        color_rgba = hex_to_rgba(color_hex, fill_alpha)
        border_rgb = hex_to_rgba(color_hex, 255)[:3]

        # Draw semi-transparent box fill
        overlay_draw.rectangle([xmin, ymin, xmax, ymax], fill=color_rgba)

        # Draw box border line with thickness
        for t in range(box_thickness):
            overlay_draw.rectangle(
                [xmin - t, ymin - t, xmax + t, ymax + t],
                outline=border_rgb
            )

        if show_labels:
            display_text = f"#{idx} {det.label}"
            
            # Measure text size
            try:
                text_bbox = font.getbbox(display_text)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except Exception:
                text_w, text_h = len(display_text) * 8, font_size

            padding = 4
            badge_w = text_w + (padding * 2)
            badge_h = text_h + (padding * 2)

            # Determine badge position (above box, or inside top if near image edge)
            badge_ymin = ymin - badge_h if ymin - badge_h >= 0 else ymin
            badge_xmin = xmin
            badge_xmax = min(width, badge_xmin + badge_w)
            badge_ymax = badge_ymin + badge_h

            # Draw background badge rectangle
            overlay_draw.rectangle(
                [badge_xmin, badge_ymin, badge_xmax, badge_ymax],
                fill=border_rgb + (230,)
            )

            # Draw label text (white or dark depending on background brightness)
            text_x = badge_xmin + padding
            text_y = badge_ymin + padding
            overlay_draw.text((text_x, text_y), display_text, fill=(255, 255, 255, 255), font=font)

    # Composite overlay onto annotated image
    final_image = Image.alpha_composite(annotated, overlay)
    return final_image.convert("RGB")
