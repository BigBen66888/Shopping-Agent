from __future__ import annotations

from typing import Any

# 模型通常只需要这些稳定字段做判断与展示；attributes 体积最大且噪声最多，
# 默认不参与工具返回，需要时由 get_product_details 单独取。
SUMMARY_FIELDS = ("product_id", "title", "brand", "category", "price", "sold_count")


def product_summary(product: Any, *, with_description: bool = False) -> dict:
    """把商品压成紧凑摘要，避免把整条记录回灌给模型（token 优化，问题 5）。"""
    record = product.to_dict() if hasattr(product, "to_dict") else dict(product)
    summary = {field: record.get(field) for field in SUMMARY_FIELDS}
    if with_description:
        description = str(record.get("description") or "")
        summary["description"] = description[:160]
    return summary


def get_product_details(product: Any, include_attributes: bool = False) -> dict:
    """返回单件商品字段。

    默认只给判断所需的核心字段 + 描述片段，避免把 0.85 KB 的 attributes
    反复回灌给模型（token 优化，问题 5）。确实需要完整规格时才传
    ``include_attributes=True``。
    """
    record = product.to_dict() if hasattr(product, "to_dict") else dict(product)
    summary = product_summary(record, with_description=True)
    summary["product_url"] = str(record.get("product_url") or "")
    if include_attributes:
        summary["attributes"] = record.get("attributes") or {}
    return summary

