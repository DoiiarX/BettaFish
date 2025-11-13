"""
Shared rendering helpers for the structured context tab in Streamlit.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_structured_context_tab(context: Dict[str, Any]):
    """Visualize the timeline, token statistics, and relations."""
    if not context or not context.get("events"):
        st.info("暂无结构化上下文数据，等待代理生成。")
        return

    _render_timeline(context)
    _render_token_statistics(context)
    _render_relations(context)
    _render_metadata(context)


def _render_timeline(context: Dict[str, Any]):
    st.subheader("时间轴")
    timeline = context.get("timeline") or []
    if not timeline:
        st.info("尚未形成有效的时间轴。")
        return
    rows: List[Dict[str, Any]] = []
    for event in timeline:
        rows.append(
            {
                "事件ID": event.get("event_id"),
                "时间": event.get("timestamp"),
                "摘要": (event.get("summary") or "")[:140],
                "核心标签": _format_labels(event.get("top_labels", [])),
                "权重": event.get("weight"),
                "来源": event.get("source"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_token_statistics(context: Dict[str, Any]):
    st.subheader("高频术语 / 标签")
    token_stats = context.get("token_statistics") or {}
    if not token_stats:
        st.info("暂无术语统计。")
        return
    tokens = sorted(
        token_stats.values(),
        key=lambda item: item.get("weight", 0),
        reverse=True,
    )[:25]
    rows = []
    for token in tokens:
        top_labels = sorted(
            token.get("labels", {}).items(), key=lambda item: item[1], reverse=True
        )[:3]
        rows.append(
            {
                "术语": token.get("token"),
                "出现次数": token.get("count"),
                "标签": ", ".join(f"{label}({score:.2f})" for label, score in top_labels),
                "最近出现时间": token.get("last_seen"),
                "示例": (token.get("examples") or [""])[0][:80],
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_relations(context: Dict[str, Any]):
    st.subheader("事件关联")
    relations = context.get("relations") or []
    if not relations:
        st.info("尚未计算出事件关联。")
        return
    for relation in relations[:50]:
        labels = ", ".join(relation.get("shared_labels", []))
        st.markdown(
            f"- `{relation.get('source_event')}` ↔ `{relation.get('target_event')}` "
            f"| 共享标签：{labels or '无'} | 权重 {relation.get('weight')}"
        )


def _render_metadata(context: Dict[str, Any]):
    metadata = context.get("metadata") or {}
    st.caption(
        f"最近更新时间：{metadata.get('updated_at', 'N/A')} "
        f"| 生成Agent：{metadata.get('agent', '未知')} "
        f"| 最近查询：{metadata.get('last_query', 'N/A')}"
    )


def _format_labels(labels: List[Dict[str, Any]]) -> str:
    return ", ".join(f"{label.get('label')}({label.get('score', 0):.2f})" for label in labels)

