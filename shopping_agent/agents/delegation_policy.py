"""Conservative, deterministic gate for independent multi-agent work.

The model still plans the individual tool calls, but it cannot delegate a
single search just because delegate tools happened to be available.
"""
from __future__ import annotations

import re


_JOINER = re.compile(r"以及|另外|同时|顺便|并且|并|和|与|还要|再|、|，|,|；|;|\band\b", re.I)
_PRODUCTS = {
    "shoes": r"鞋子|运动鞋|球鞋|皮鞋|跑鞋|靴子|\bshoes?\b|\bsneakers?\b",
    "headphones": r"耳机|耳麦|\bheadphones?\b|\bearbuds?\b",
    "mouse": r"鼠标|\bmice\b|\bmouse\b",
    "keyboard": r"键盘|\bkeyboard\b",
    "bag": r"背包|书包|手提包|行李箱|\bbackpacks?\b|\bbags?\b",
    "phone": r"手机|\bphones?\b|\biphone\b",
    "computer": r"电脑|笔记本电脑|\blaptops?\b|\bcomputers?\b",
    "bottle": r"水杯|水壶|杯子|\bbottles?\b",
    "monitor": r"显示器|\bmonitors?\b",
    "battery": r"电池|充电宝|\bbatter(?:y|ies)\b",
    "clothing": r"衣服|服装|外套|裤子|\bclothes\b|\bjackets?\b",
}
_STATES = {
    "cart": r"购物车|购物袋|\bcart\b",
    "orders": r"订单|\borders?\b",
    "favorites": r"收藏夹|我的收藏|收藏列表|\bfavorites?\b",
    "memory": r"长期记忆|偏好记忆|我的记忆|\bmemor(?:y|ies)\b",
}
_DEPENDENT_WRITE = re.compile(
    r"(?:把|将|买|下单).{0,15}(?:它|这(?:个|件|款)|第[一二三四五六七八九十\d]+(?:个|件)|"
    r"找到的|搜到的|推荐的|上述|刚才的).{0,18}(?:加|放|购|买|下单)|"
    r"(?:把|将).{0,12}(?:它|这(?:个|件|款)|第[一二三四五六七八九十\d]+(?:个|件)|"
    r"找到的|搜到的|推荐的).{0,15}购物车"
)
_OBJECT_PAIR = re.compile(
    r"(?:找|看看|搜索|推荐|挑选|选购|买).{0,18}?"
    r"([\u4e00-\u9fff]{2,8})\s*(?:和|与|以及|及|、)\s*([\u4e00-\u9fff]{2,8})"
)
_NON_PRODUCT = re.compile(r"预算|价格|尺码|颜色|品牌|用途|元|码|购物车|订单|收藏|记忆")
_OPEN_ENDED_RESEARCH = re.compile(
    r"(?:看看|了解|推荐|研究|列出|规划).{0,14}(?:需要|该|应该|适合)?(?:买|准备|带)(?:些|点)?什么|"
    r"(?:需要|该|应该)(?:买|准备|带)(?:些|点)?什么|"
    r"(?:买|准备)(?:些|点)?什么(?:商品|东西|装备|用品)?"
)
_RESEARCH_VERB = re.compile(r"看看|找|搜索|推荐|挑选|选购|研究|需要买|买什么")
_STATE_ONLY = re.compile(r"购物车|购物袋|订单|收藏|记忆")


def _has_product_research(text: str, products: set[str]) -> bool:
    if products:
        return bool(_RESEARCH_VERB.search(text))
    # An activity such as 登山/露营 can imply product research without naming a category.
    return bool(_OPEN_ENDED_RESEARCH.search(text)) and not bool(
        _STATE_ONLY.fullmatch(text.strip())
    )


def _has_two_product_objects(text: str) -> bool:
    """Recognize short paired product nouns beyond the common category list."""
    match = _OBJECT_PAIR.search(text)
    if not match:
        return False
    return all(not _NON_PRODUCT.search(part) for part in match.groups())


def delegation_triggers(query: str) -> tuple[str, ...]:
    """Return only hard, independent-task triggers present in this turn.

    1. Two distinct product types to research.
    2. Product research plus an independent existing-state task.
    3. Two distinct existing-state tasks.

    A search followed by an action on *that search result* stays with MainAgent.
    """
    text = (query or "").strip().casefold()
    if not text or not _JOINER.search(text):
        return ()
    products = {name for name, pattern in _PRODUCTS.items() if re.search(pattern, text, re.I)}
    states = {name for name, pattern in _STATES.items() if re.search(pattern, text, re.I)}
    reasons: list[str] = []
    if len(products) >= 2 or _has_two_product_objects(text):
        reasons.append("multiple_product_searches")
    if _has_product_research(text, products) and states and not _DEPENDENT_WRITE.search(text):
        reasons.append("product_and_independent_state")
    if len(states) >= 2:
        reasons.append("multiple_state_tasks")
    return tuple(reasons)
