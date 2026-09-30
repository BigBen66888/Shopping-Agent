"""Conservative, local slot matching for durable shopping-memory updates.

Free-form facts are retained unless both their subject and preference facet can
be identified. This avoids deleting unrelated preferences to improve a score.
"""
from __future__ import annotations

import re

SUBJECTS = (
    ("背包", "backpack"), ("书包", "backpack"), ("行李箱", "luggage"),
    ("鼠标", "mouse"), ("耳机", "headphones"), ("键盘", "keyboard"),
    ("运动鞋", "shoes"), ("鞋", "shoes"), ("水杯", "bottle"),
    ("显示器", "monitor"), ("电脑", "computer"), ("手机", "phone"),
    ("衣服", "clothing"), ("服装", "clothing"), ("包", "bag"),
    ("backpack", "backpack"), ("mouse", "mouse"), ("headphone", "headphones"),
    ("keyboard", "keyboard"), ("shoe", "shoes"),
)
COLORS = {"黑色": "black", "白色": "white", "红色": "red", "蓝色": "blue",
          "绿色": "green", "灰色": "grey", "黄色": "yellow", "粉色": "pink",
          "棕色": "brown", "紫色": "purple", "black": "black", "white": "white",
          "red": "red", "blue": "blue", "green": "green", "grey": "grey",
          "gray": "grey", "yellow": "yellow", "pink": "pink", "brown": "brown"}
BRANDS = {"罗技": "logitech", "logitech": "logitech", "索尼": "sony", "sony": "sony",
          "雷蛇": "razer", "razer": "razer", "苹果": "apple", "apple": "apple",
          "华为": "huawei", "huawei": "huawei", "小米": "xiaomi", "xiaomi": "xiaomi",
          "联想": "lenovo", "lenovo": "lenovo", "戴尔": "dell", "dell": "dell",
          "惠普": "hp", "hp": "hp"}


def _match_value(text: str, values: dict[str, str]) -> str | None:
    found: list[tuple[int, str, bool, bool]] = []
    for token in sorted(values, key=len, reverse=True):
        for match in re.finditer(r"(?<![a-z])" + re.escape(token) + r"(?![a-z])", text):
            preceding = text[max(0, match.start() - 10):match.start()]
            negative = bool(re.search(r"(?:不再|不要|不买|不选|放弃|避开|不喜欢)[^，,。]*$", preceding))
            changed = bool(re.search(r"(?:改为|改成|换成|现在|以后|优先|只买)[^，,。]*$", preceding))
            found.append((match.start(), values[token], negative, changed))
    if not found:
        return None
    positive = [item for item in found if not item[2]]
    if positive:
        chosen = max(positive, key=lambda item: (item[3], item[0]))
        return chosen[1]
    return "not:" + max(found, key=lambda item: item[0])[1]


def memory_slot(kind: str, content: str) -> tuple[str, str, str] | None:
    """Return (subject, facet, value), or None when safe matching is impossible."""
    text = content.casefold()
    subject = next((canonical for token, canonical in SUBJECTS if token in text), "general")
    color = _match_value(text, COLORS)
    if color and kind in {"attribute", "constraint"}:
        return subject, "color", color
    if kind == "brand":
        brand = _match_value(text, BRANDS)
        return (subject, "brand", brand) if brand else None
    if kind == "budget":
        amount = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:元|比索|php|peso)", text)
        return (subject, "budget", amount.group(1)) if amount else None
    if kind in {"attribute", "constraint"}:
        size = re.search(r"(?<!\d)(\d{2})\s*码", text)
        if size:
            return subject, "shoe_size", size.group(1)
        if "无线" in text or "wireless" in text:
            return subject, "connectivity", "wireless"
        if "有线" in text or "wired" in text:
            return subject, "connectivity", "wired"
    return None
