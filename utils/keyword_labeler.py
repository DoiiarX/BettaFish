"""
Offline keyword labeling powered by a lightweight keyword dictionary.

This version avoids remote LLM calls by matching user-provided categories
against the text using substring checks plus optional rapidfuzz scoring.
"""

from __future__ import annotations

import subprocess
import sys
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger


class _FallbackFuzz:
    """Simplified fuzzy scorer when rapidfuzz is unavailable."""

    @staticmethod
    def partial_ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        matcher = SequenceMatcher(None, a, b)
        return matcher.ratio() * 100


_ATTEMPTED_AUTOINSTALL = False


def _load_rapidfuzz():
    global _ATTEMPTED_AUTOINSTALL
    try:
        from rapidfuzz import fuzz  # type: ignore
        return fuzz
    except ImportError:  # pragma: no cover - optional dependency
        if not _ATTEMPTED_AUTOINSTALL:
            logger.warning("rapidfuzz 未安装，尝试自动安装...")
            _ATTEMPTED_AUTOINSTALL = True
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "rapidfuzz"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                from rapidfuzz import fuzz  # type: ignore

                logger.info("rapidfuzz 安装完成。")
                return fuzz
            except Exception as exc:
                logger.warning(f"rapidfuzz 安装失败，将使用简化模糊匹配: {exc}")
        return _FallbackFuzz()


fuzz = _load_rapidfuzz()


CATEGORY_KEYWORDS: List[Dict[str, Any]] = [
    {"zh": "手工", "en": "Handcraft", "keywords": ["手工", "手作", "DIY", "编织", "工艺", "雕刻", "缝纫"]},
    {"zh": "美食", "en": "Food", "keywords": ["美食", "吃播", "餐厅", "料理", "烘焙", "食谱", "美味", "下饭"]},
    {"zh": "小剧场", "en": "Mini Drama", "keywords": ["小剧场", "短剧", "微短剧", "情景剧", "剧情向", "反转剧"]},
    {"zh": "旅游出行", "en": "Travel", "keywords": ["旅游", "旅行", "出行", "景点", "度假", "徒步", "自驾", "打卡"]},
    {"zh": "三农", "en": "Agri Rural", "keywords": ["三农", "农业", "农民", "乡村", "农场", "种植", "乡间"]},
    {"zh": "动物", "en": "Animals", "keywords": ["动物", "萌宠", "宠物", "小猫", "小狗", "铲屎官", "野生动物", "动物园"]},
    {"zh": "汽车", "en": "Automotive", "keywords": ["汽车", "跑车", "新能源车", "车评", "驾车", "车展", "车辆"]},
    {
        "zh": "时尚美妆",
        "en": "Fashion & Beauty",
        "keywords": ["时尚", "美妆", "穿搭", "口红", "化妆", "护肤", "lookbook"],
    },
    {
        "zh": "家装房产",
        "en": "Home & Real Estate",
        "keywords": ["家装", "装修", "家居", "房产", "软装", "样板间", "租房"],
    },
    {"zh": "户外潮流", "en": "Outdoor Trends", "keywords": ["户外", "露营", "机能风", "潮流", "冲锋衣", "徒步鞋"]},
    {"zh": "健身", "en": "Fitness", "keywords": ["健身", "力量", "增肌", "塑形", "撸铁", "HIIT", "瑜伽"]},
    {
        "zh": "体育运动",
        "en": "Sports",
        "keywords": ["体育", "运动", "篮球", "足球", "网球", "羽毛球", "马拉松", "滑雪"],
    },
    {"zh": "鬼畜", "en": "Guichu Remix", "keywords": ["鬼畜", "鬼畜剪辑", "鬼畜混剪"]},
    {"zh": "游戏", "en": "Gaming", "keywords": ["游戏", "手游", "主机", "电竞", "开荒", "升级", "联机"]},
    {"zh": "资讯", "en": "News", "keywords": ["资讯", "新闻", "快讯", "播报", "头条"]},
    {"zh": "知识", "en": "Knowledge", "keywords": ["知识", "科普", "冷知识", "讲解", "百科"]},
    {
        "zh": "人工智能",
        "en": "Artificial Intelligence",
        "keywords": ["人工智能", "AI", "大模型", "机器学习", "智能体", "算法"],
    },
    {
        "zh": "科技数码",
        "en": "Tech & Gadgets",
        "keywords": ["科技", "数码", "手机", "笔记本", "评测", "芯片", "电子产品"],
    },
    {"zh": "影视", "en": "Film & TV", "keywords": ["影视", "电影", "电视剧", "影评", "镜头", "剧透"]},
    {"zh": "娱乐", "en": "Entertainment", "keywords": ["娱乐", "明星", "八卦", "综艺", "粉丝"]},
    {"zh": "音乐", "en": "Music", "keywords": ["音乐", "歌曲", "翻唱", "乐队", "吉他", "钢琴", "演奏"]},
    {"zh": "舞蹈", "en": "Dance", "keywords": ["舞蹈", "跳舞", "编舞", "舞台", "舞者"]},
    {"zh": "动画", "en": "Animation", "keywords": ["动画", "动漫", "番剧", "国漫", "追番"]},
    {"zh": "绘画", "en": "Painting", "keywords": ["绘画", "画画", "插画", "速写", "水彩", "临摹"]},
    {"zh": "亲子", "en": "Parenting", "keywords": ["亲子", "宝妈", "宝爸", "育儿", "早教", "家庭教育"]},
    {"zh": "健康", "en": "Health", "keywords": ["健康", "养生", "体检", "营养", "理疗", "康复"]},
    {"zh": "情感", "en": "Emotions", "keywords": ["情感", "感情", "恋爱", "婚姻", "告白", "分手", "心理"]},
    {"zh": "vlog", "en": "Vlog", "keywords": ["vlog", "生活vlog", "日常记录", "日常分享"]},
    {"zh": "生活兴趣", "en": "Lifestyle Hobbies", "keywords": ["生活兴趣", "兴趣", "手帐", "收藏", "插花", "陶艺"]},
    {"zh": "生活经验", "en": "Life Tips", "keywords": ["生活经验", "经验", "避坑", "心得", "实用技巧"]},
]

LABEL_SCHEMA: List[str] = [
    *(category["zh"] for category in CATEGORY_KEYWORDS),
    *(category["en"] for category in CATEGORY_KEYWORDS),
]
DEFAULT_FUZZY_THRESHOLD = 0.78


class KeywordLabelingClient:
    """Performs keyword-based labeling without external model calls."""

    def __init__(self, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD):
        self.fuzzy_threshold = fuzzy_threshold
        self._keyword_entries: List[Tuple[str, str, str]] = self._build_keyword_entries()

    def label_terms(
        self, text: str, *, topic: str = "", max_terms: int = 8
    ) -> List[Dict[str, List[Dict[str, float]]]]:
        if not text:
            return []

        snippet = text[:2000]
        matches = self._detect_matches(snippet)
        annotations = []
        seen_terms = set()

        for term, label_zh, label_en, score in matches:
            term_key = term.lower()
            if term_key in seen_terms:
                continue
            per_label_score = round(score / 2, 3)
            annotations.append(
                {
                    "term": term,
                    "labels": [
                        {
                            "label": label_zh,
                            "score": per_label_score,
                        },
                        {
                            "label": label_en,
                            "score": per_label_score,
                        },
                    ],
                }
            )
            seen_terms.add(term_key)
            if len(annotations) >= max_terms:
                break

        return annotations

    def _build_keyword_entries(self) -> List[Tuple[str, str, str]]:
        entries: List[Tuple[str, str, str]] = []
        for category in CATEGORY_KEYWORDS:
            for keyword in category["keywords"]:
                entries.append((keyword, category["zh"], category["en"]))
        return entries

    def _detect_matches(self, text: str) -> List[Tuple[str, str, str, float]]:
        direct = self._direct_matches(text)
        fuzzy_matches = self._fuzzy_matches(text, {kw.lower() for kw, _, _, _ in direct})
        combined = direct + fuzzy_matches
        return sorted(combined, key=lambda item: item[3], reverse=True)

    def _direct_matches(self, text: str) -> List[Tuple[str, str, str, float]]:
        matches: List[Tuple[str, str, str, float]] = []
        lowered = text.lower()
        for keyword, label_zh, label_en in self._keyword_entries:
            keyword_lower = keyword.lower()
            idx = lowered.find(keyword_lower)
            if idx == -1:
                continue
            term = text[idx : idx + len(keyword)]
            matches.append((term, label_zh, label_en, 0.95))
        return matches

    def _fuzzy_matches(
        self, text: str, existing_terms: Sequence[str]
    ) -> List[Tuple[str, str, str, float]]:
        matches: List[Tuple[str, str, str, float]] = []
        lowered = text.lower()
        for keyword, label_zh, label_en in self._keyword_entries:
            keyword_lower = keyword.lower()
            if keyword_lower in existing_terms:
                continue
            score = fuzz.partial_ratio(lowered, keyword_lower) / 100.0
            if score >= self.fuzzy_threshold:
                matches.append((keyword, label_zh, label_en, score))
        return matches


@lru_cache(maxsize=1)
def get_keyword_labeler() -> Optional[KeywordLabelingClient]:
    return KeywordLabelingClient()
