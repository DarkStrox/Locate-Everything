import re
from typing import List, Dict, Any, Tuple

class DetectionBox:
    def __init__(self, label: str, ymin_norm: float, xmin_norm: float, ymax_norm: float, xmax_norm: float):
        self.label = label.strip() if label else "object"
        self.ymin_norm = float(ymin_norm)
        self.xmin_norm = float(xmin_norm)
        self.ymax_norm = float(ymax_norm)
        self.xmax_norm = float(xmax_norm)

    def to_pixel_box(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Converts normalized 0-1000 coordinates to pixel bounding box (xmin, ymin, xmax, ymax).
        Clamps values to image boundary.
        """
        # Determine scale factor (if values are 0-1000 or 0-1)
        max_val = max(self.ymin_norm, self.xmin_norm, self.ymax_norm, self.xmax_norm)
        scale = 1000.0 if max_val > 1.5 else 1.0

        ymin_px = int(round((self.ymin_norm / scale) * img_height))
        xmin_px = int(round((self.xmin_norm / scale) * img_width))
        ymax_px = int(round((self.ymax_norm / scale) * img_height))
        xmax_px = int(round((self.xmax_norm / scale) * img_width))

        # Clamp to bounds
        xmin = max(0, min(xmin_px, img_width - 1))
        ymin = max(0, min(ymin_px, img_height - 1))
        xmax = max(xmin + 1, min(xmax_px, img_width))
        ymax = max(ymin + 1, min(ymax_px, img_height))

        return xmin, ymin, xmax, ymax

    def to_dict(self, img_width: int = 0, img_height: int = 0) -> Dict[str, Any]:
        data = {
            "label": self.label,
            "norm_box": [self.ymin_norm, self.xmin_norm, self.ymax_norm, self.xmax_norm]
        }
        if img_width > 0 and img_height > 0:
            xmin, ymin, xmax, ymax = self.to_pixel_box(img_width, img_height)
            data["pixel_box"] = [xmin, ymin, xmax, ymax]
            area_pct = ((xmax - xmin) * (ymax - ymin)) / float(img_width * img_height) * 100.0
            data["area_pct"] = round(area_pct, 2)
        return data


def parse_grounding_response(response_text: str, default_label: str = "detected object") -> List[DetectionBox]:
    """
    Parses vision-grounding output from models like LocateAnything, LocateEverything, Qwen2-VL.
    Tokenizes <ref>...</ref> and <box>...</box> tags in order.
    Supports multiple <box> tags associated with a single <ref> tag.
    """
    detections = []
    if not response_text or not response_text.strip():
        return detections

    # Tokenize tags in order: <ref>...</ref> or <box>...</box>
    token_pattern = r'(<ref>.*?</ref>|<box>.*?</box>)'
    tokens = re.findall(token_pattern, response_text, re.DOTALL | re.IGNORECASE)

    current_label = default_label

    for tok in tokens:
        if tok.lower().startswith('<ref>'):
            m = re.match(r'<ref>(.*?)</ref>', tok, re.DOTALL | re.IGNORECASE)
            if m and m.group(1).strip():
                current_label = m.group(1).strip()
        elif tok.lower().startswith('<box>'):
            m = re.match(r'<box>(.*?)</box>', tok, re.DOTALL | re.IGNORECASE)
            if m:
                content = m.group(1)
                nums = [float(x) for x in re.findall(r'[\d.]+', content)]
                if len(nums) == 4:
                    detections.append(DetectionBox(current_label, nums[0], nums[1], nums[2], nums[3]))

    if detections:
        return detections

    # Fallback Pattern: Generic bracket numbers [y1, x1, y2, x2] outside <box> tags
    p_fallback = r"\[(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})\]"
    matches_fb = re.findall(p_fallback, response_text)
    for idx, (y1, x1, y2, x2) in enumerate(matches_fb, start=1):
        lbl = f"{default_label} {idx}"
        detections.append(DetectionBox(lbl, float(y1), float(x1), float(y2), float(x2)))

    return detections
