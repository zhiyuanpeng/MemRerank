"""
Reranking reward with deterministic memory-quality shaping v2.

This reward keeps the same VERL entrypoints as rlvr_reward_fast_margin.py:
compute_score_w and compute_score_wc.

Main differences:
- Evidence fields are stripped before memory is passed to the reward ranker.
- M_WC requires a non-empty within-category memory, but cross memory may be empty.
- A small query-independent quality term is added after the reranking utility.
- The final reward is pos_freq + quality_weight * quality_score.
- No margin term is used.

The quality term uses only generated memory, target category, and purchase
history. It deliberately does not use the current query or candidate products.
"""

import logging
import re
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.reward.llm_ranker_rd import RewardRanker


PLACEHOLDER_EVIDENCE_TERMS = (
    "within-category history",
    "cross-category history",
    "history metadata",
    "input history",
    "metadata only",
    "n/a",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "user",
    "with",
}

GENERIC_MEMORY_TOKENS = {
    "aspect",
    "based",
    "category",
    "cross",
    "general",
    "history",
    "memory",
    "need",
    "needs",
    "openness",
    "pattern",
    "prefer",
    "preference",
    "preferences",
    "prefers",
    "product",
    "products",
    "shopping",
    "tendency",
    "towards",
    "user",
    "wants",
}

BOILERPLATE_PHRASES = (
    "tendency towards",
    "preference for preference",
    "need for need",
    "openness to",
    "durable and reliable",
    "high quality",
    "portable gear",
    "sustainable brands",
    "general shopping style",
)

GENERIC_CROSS_PHRASES = (
    "sustainable brands",
    "durable and reliable",
    "portable gear",
    "portable devices",
    "high quality",
    "long-lasting gear",
    "long-lasting products",
)


QUALITY_DETAIL_KEYS = (
    "quality_score",
    "ranking_memory_words",
    "within_memory_words",
    "cross_memory_words",
    "history_words",
    "bullet_count",
    "evidence_count",
    "copied_evidence_count",
    "fake_evidence_count",
    "unsupported_evidence_count",
    "duplicate_aspect_count",
    "short_preference_line_count",
    "within_history_overlap_count",
    "long_within_copy",
    "long_cross_copy",
    "cross_memory_empty",
    "cross_too_long",
    "cross_category_leakage_count",
    "supported_within_line_count",
    "unsupported_within_line_count",
    "wrong_side_within_line_count",
    "supported_cross_line_count",
    "unsupported_cross_line_count",
    "wrong_side_cross_line_count",
    "boilerplate_line_count",
    "aspect_pref_duplicate_count",
    "generic_cross_line_count",
    "weak_specificity_line_count",
    "explicit_bullet_count",
)


def default_quality_details() -> Dict[str, float]:
    """Return the full quality metric schema for malformed generations."""
    return {key: 0.0 for key in QUALITY_DETAIL_KEYS}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9_+.-]*", (text or "").lower())


def _content_tokens(text: str) -> List[str]:
    return [
        token
        for token in _tokens(text)
        if len(token) >= 3 and token not in STOPWORDS and token not in GENERIC_MEMORY_TOKENS
    ]


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _extract_block(solution: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", solution or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def _strip_evidence_from_line(line: str) -> str:
    match = re.search(r"\bEvidence\s*:", line, flags=re.IGNORECASE)
    if not match:
        return line.rstrip()
    return line[: match.start()].rstrip(" ;.").rstrip()


def strip_evidence_fields(memory_text: str) -> str:
    """Remove Evidence fields before memory is sent to the reward ranker."""
    lines = [_strip_evidence_from_line(line) for line in (memory_text or "").splitlines()]
    return "\n".join(line for line in lines if line.strip()).strip()


def extract_evidence_snippets(memory_text: str) -> List[str]:
    snippets: List[str] = []
    for line in (memory_text or "").splitlines():
        match = re.search(r"\bEvidence\s*:\s*(.*)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        snippets.extend(re.findall(r'"([^"]+)"', match.group(1)))
    return snippets


def _explicit_bullet_lines(memory_text: str) -> List[str]:
    return [line.strip() for line in (memory_text or "").splitlines() if line.strip().startswith("-")]


def _memory_lines(memory_text: str) -> List[str]:
    """Return scorable memory lines for both M_WC_v3 bullets and baseline facts.

    Training still prefers M_WC_v3-style bullets, but the offline sanity check
    compares against heterogeneous baselines such as Mem0 and MR.Rec. This
    splitter keeps the reward category-general without depending on a specific
    baseline format.
    """
    lines: List[str] = []
    for raw_line in (memory_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            lines.append(line)
            continue
        parts = [part.strip() for part in re.split(r"\s*\|\s*", line) if part.strip()]
        if len(parts) > 1:
            lines.extend(f"- {part}" for part in parts)
        else:
            numbered = re.sub(r"^\d+\.\s*", "", line).strip()
            if numbered:
                lines.append(f"- {numbered}")
    return lines


def _aspect_label(line: str) -> str:
    match = re.match(r"-\s*(?:Aspect|Pattern)\s*:\s*([^;:]+)", line, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.match(r"-\s*([^:;]{1,80})\s*:", line)
    if match:
        return match.group(1).strip()
    return ""


def _line_parts(line: str) -> Tuple[str, str]:
    match = re.match(
        r"-\s*(?:Aspect|Pattern)\s*:\s*([^;:]+)\s*;\s*Preference\s*:\s*(.*)$",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return _aspect_label(line), line


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", (label or "").lower()).strip()


def _normalize_category(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (category or "").lower()).strip()


def _history_sides(data_source: Dict[str, Any]) -> Tuple[str, str, str, List[str]]:
    history = data_source.get("history") or {}
    target_category = _normalize_category(data_source.get("category_name", ""))
    within_chunks: List[str] = []
    cross_chunks: List[str] = []
    cross_categories: List[str] = []

    if not isinstance(history, dict):
        return "", "", "", []

    metas = history.get("metas") or history.get("metadata") or []
    reviews = history.get("reviews") or []
    categories = history.get("categories") or []

    n = max(len(metas) if isinstance(metas, list) else 0, len(categories) if isinstance(categories, list) else 0)
    for i in range(n):
        meta = str(metas[i]) if isinstance(metas, list) and i < len(metas) else ""
        review = str(reviews[i]) if isinstance(reviews, list) and i < len(reviews) else ""
        category = str(categories[i]) if isinstance(categories, list) and i < len(categories) else ""
        category_norm = _normalize_category(category)
        chunk = " ".join(part for part in (meta, review) if part).strip()
        if not chunk:
            continue
        if category_norm == target_category:
            within_chunks.append(chunk)
        else:
            cross_chunks.append(chunk)
            if category_norm and category_norm not in cross_categories:
                cross_categories.append(category_norm)

    # Fallback for older data without category-aligned history fields.
    if not within_chunks and isinstance(metas, list):
        within_chunks = [str(x) for x in metas if str(x).strip()]
    all_text = "\n".join(within_chunks + cross_chunks).lower()
    return "\n".join(within_chunks).lower(), "\n".join(cross_chunks).lower(), all_text, cross_categories


def _count_copied_evidence(snippets: List[str], history_text: str) -> int:
    return sum(1 for snippet in snippets if snippet and snippet.lower() in history_text)


def _count_placeholder_evidence(snippets: List[str]) -> int:
    return sum(
        1 for snippet in snippets if any(term in snippet.lower() for term in PLACEHOLDER_EVIDENCE_TERMS)
    )


def _max_ngram_overlap(a_text: str, b_text: str, n: int = 6) -> int:
    a_tokens = _content_tokens(a_text)
    b_tokens = _content_tokens(b_text)
    if len(a_tokens) < n or len(b_tokens) < n:
        return 0
    b_ngrams = {tuple(b_tokens[i : i + n]) for i in range(len(b_tokens) - n + 1)}
    best = 0
    run = 0
    for i in range(len(a_tokens) - n + 1):
        if tuple(a_tokens[i : i + n]) in b_ngrams:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best + n - 1 if best else 0


def _content_overlap(memory_text: str, history_text: str) -> int:
    mem = set(_content_tokens(memory_text))
    hist = set(_content_tokens(history_text))
    return len(mem & hist)


def _cross_category_name_leakage(cross_memory: str, cross_categories: List[str]) -> int:
    memory_norm = _normalize_category(cross_memory)
    count = 0
    for category in cross_categories:
        if category and category in memory_norm:
            count += 1
    return count


def _count_phrase_matches(text: str, phrases: Tuple[str, ...]) -> int:
    text_norm = (text or "").lower()
    return sum(1 for phrase in phrases if phrase in text_norm)


def _jaccard(a_tokens: List[str], b_tokens: List[str]) -> float:
    a = set(a_tokens)
    b = set(b_tokens)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _is_aspect_preference_duplicate(aspect: str, preference: str) -> bool:
    aspect_tokens = _content_tokens(aspect)
    pref_tokens = _content_tokens(preference)
    if not aspect_tokens or not pref_tokens:
        return False
    a = set(aspect_tokens)
    p = set(pref_tokens)
    if _jaccard(aspect_tokens, pref_tokens) >= 0.75:
        return True
    smaller = min(len(a), len(p))
    if smaller <= 2 and len(a & p) == smaller:
        return True
    return False


def _line_quality_stats(
    line: str,
    *,
    primary_tokens: set,
    secondary_tokens: set,
) -> Dict[str, float]:
    aspect, preference = _line_parts(line)
    line_tokens = set(_content_tokens(f"{aspect} {preference}"))
    primary_overlap = len(line_tokens & primary_tokens)
    secondary_overlap = len(line_tokens & secondary_tokens)
    specificity = len(line_tokens)
    boilerplate = _count_phrase_matches(line, BOILERPLATE_PHRASES)
    duplicate = 1 if _is_aspect_preference_duplicate(aspect, preference) else 0
    weak_specificity = 1 if specificity < 3 else 0
    supported = 1 if primary_overlap >= 1 and specificity >= 2 else 0
    unsupported = 1 if primary_overlap == 0 and secondary_overlap == 0 and specificity < 3 else 0
    wrong_side = 1 if secondary_overlap >= primary_overlap + 2 and primary_overlap == 0 else 0
    return {
        "primary_overlap": float(primary_overlap),
        "secondary_overlap": float(secondary_overlap),
        "specificity": float(specificity),
        "boilerplate": float(boilerplate),
        "duplicate": float(duplicate),
        "weak_specificity": float(weak_specificity),
        "supported": float(supported),
        "unsupported": float(unsupported),
        "wrong_side": float(wrong_side),
    }


def score_memory_quality(solution: str, data_source: Dict[str, Any], requires_cross: bool = True) -> Dict[str, float]:
    """
    Deterministic quality shaping for query-independent preference memory v2.

    The output is intentionally small, roughly in [-0.20, 0.20], so reranking
    utility remains the dominant training signal. The score is line-based and
    rewards grounded, side-correct, non-boilerplate memories.
    """
    within_raw = _extract_block(solution, "within_memory")
    cross_raw = _extract_block(solution, "cross_memory") if requires_cross else ""
    within_rank = strip_evidence_fields(within_raw)
    cross_rank = strip_evidence_fields(cross_raw)
    ranking_memory = "\n".join(x for x in (within_rank, cross_rank) if x).strip()

    within_hist, cross_hist, all_hist, cross_categories = _history_sides(data_source)
    within_words = _word_count(within_rank)
    cross_words = _word_count(cross_rank)
    total_words = _word_count(ranking_memory)
    history_words = max(1, _word_count(all_hist))

    within_evidence = extract_evidence_snippets(within_raw)
    cross_evidence = extract_evidence_snippets(cross_raw)
    all_evidence = within_evidence + cross_evidence
    fake_evidence = _count_placeholder_evidence(all_evidence)
    copied_within_evidence = _count_copied_evidence(within_evidence, within_hist or all_hist)
    copied_cross_evidence = _count_copied_evidence(cross_evidence, cross_hist)
    copied_evidence = copied_within_evidence + copied_cross_evidence
    unsupported_evidence = max(0, len(all_evidence) - copied_evidence - fake_evidence)

    within_lines = _memory_lines(within_rank)
    cross_lines = _memory_lines(cross_rank)
    explicit_bullets = len(_explicit_bullet_lines(within_rank)) + len(_explicit_bullet_lines(cross_rank))
    lines = within_lines + cross_lines
    labels = [_normalize_label(_aspect_label(line)) for line in lines if _aspect_label(line)]
    duplicate_labels = max(0, len(labels) - len(set(labels)))
    short_preference_lines = sum(1 for line in lines if len(_content_tokens(line)) < 3)

    within_tokens = set(_content_tokens(within_hist))
    cross_tokens = set(_content_tokens(cross_hist))
    within_overlap = _content_overlap(within_rank, within_hist)
    long_within_copy = _max_ngram_overlap(within_rank, within_hist)
    long_cross_copy = _max_ngram_overlap(cross_rank, cross_hist)
    cross_empty = 1 if not cross_rank.strip() else 0
    cross_too_long = 1 if cross_words > max(45, int(1.2 * max(1, within_words))) else 0
    cross_category_leakage = _cross_category_name_leakage(cross_rank, cross_categories)

    supported_within = 0
    unsupported_within = 0
    wrong_side_within = 0
    supported_cross = 0
    unsupported_cross = 0
    wrong_side_cross = 0
    boilerplate_lines = 0
    aspect_pref_duplicates = 0
    generic_cross_lines = 0
    weak_specificity_lines = 0

    quality = 0.0
    if within_words > 0 and within_lines:
        quality += 0.025

    if 2 <= len(lines) <= 7:
        quality += 0.02
    elif 8 <= len(lines) <= 10:
        quality += 0.005
    elif len(lines) > 10:
        quality -= min(0.05, 0.01 * (len(lines) - 10))

    if 25 <= total_words <= 120:
        quality += 0.025
    elif 121 <= total_words <= 160:
        quality += 0.005
    elif total_words < 15:
        quality -= 0.04
    elif total_words > 180:
        quality -= min(0.12, 0.025 * ((total_words - 180) / 50.0))

    if total_words < 0.5 * history_words and total_words >= 20:
        quality += 0.01

    for line in within_lines:
        stats = _line_quality_stats(line, primary_tokens=within_tokens, secondary_tokens=cross_tokens)
        supported_within += int(stats["supported"])
        unsupported_within += int(stats["unsupported"])
        wrong_side_within += int(stats["wrong_side"])
        boilerplate_lines += int(stats["boilerplate"])
        aspect_pref_duplicates += int(stats["duplicate"])
        weak_specificity_lines += int(stats["weak_specificity"])

        if stats["supported"]:
            quality += 0.014
        if stats["specificity"] >= 3:
            quality += 0.008
        if stats["unsupported"]:
            quality -= 0.01
        if stats["wrong_side"]:
            quality -= 0.03
        if stats["boilerplate"]:
            quality -= min(0.025, 0.0125 * stats["boilerplate"])
        if stats["duplicate"]:
            quality -= 0.02
        if stats["weak_specificity"]:
            quality -= 0.01

    if cross_empty:
        quality += 0.012
    else:
        if cross_words <= within_words:
            quality += 0.006
        for line in cross_lines:
            stats = _line_quality_stats(line, primary_tokens=cross_tokens, secondary_tokens=within_tokens)
            generic_cross = 1 if _count_phrase_matches(line, GENERIC_CROSS_PHRASES) and stats["primary_overlap"] < 2 else 0
            supported_cross += int(stats["supported"])
            unsupported_cross += int(stats["unsupported"])
            wrong_side_cross += int(stats["wrong_side"])
            boilerplate_lines += int(stats["boilerplate"])
            aspect_pref_duplicates += int(stats["duplicate"])
            generic_cross_lines += generic_cross
            weak_specificity_lines += int(stats["weak_specificity"])

            if stats["supported"]:
                quality += 0.008
            if stats["specificity"] >= 3 and stats["primary_overlap"] >= 1:
                quality += 0.004
            if stats["unsupported"]:
                quality -= 0.01
            if stats["wrong_side"]:
                quality -= 0.018
            if generic_cross:
                quality -= 0.028
            if stats["boilerplate"]:
                quality -= min(0.025, 0.0125 * stats["boilerplate"])
            if stats["duplicate"]:
                quality -= 0.018
            if stats["weak_specificity"]:
                quality -= 0.01

        if cross_lines and supported_cross == 0:
            quality -= 0.012

    quality += min(0.012, 0.003 * copied_evidence)

    quality -= min(0.12, 0.04 * fake_evidence)
    quality -= min(0.02, 0.004 * unsupported_evidence)
    quality -= min(0.06, 0.02 * duplicate_labels)
    quality -= min(0.05, 0.012 * short_preference_lines)
    quality -= min(0.05, 0.02 * cross_category_leakage)
    if cross_too_long:
        quality -= 0.03
    if long_within_copy >= 8:
        quality -= min(0.04, 0.01 * (long_within_copy - 7))
    if long_cross_copy >= 8:
        quality -= min(0.04, 0.01 * (long_cross_copy - 7))
    if all_evidence and copied_evidence == 0:
        quality -= 0.04
    if within_lines and supported_within == 0:
        quality -= 0.015
    if not within_lines:
        quality -= 0.06
    quality = max(-0.20, min(0.20, quality))

    return {
        "quality_score": round(quality, 4),
        "ranking_memory_words": float(total_words),
        "within_memory_words": float(within_words),
        "cross_memory_words": float(cross_words),
        "history_words": float(history_words),
        "bullet_count": float(len(lines)),
        "evidence_count": float(len(all_evidence)),
        "copied_evidence_count": float(copied_evidence),
        "fake_evidence_count": float(fake_evidence),
        "unsupported_evidence_count": float(unsupported_evidence),
        "duplicate_aspect_count": float(duplicate_labels),
        "short_preference_line_count": float(short_preference_lines),
        "within_history_overlap_count": float(within_overlap),
        "long_within_copy": float(long_within_copy),
        "long_cross_copy": float(long_cross_copy),
        "cross_memory_empty": float(cross_empty),
        "cross_too_long": float(cross_too_long),
        "cross_category_leakage_count": float(cross_category_leakage),
        "supported_within_line_count": float(supported_within),
        "unsupported_within_line_count": float(unsupported_within),
        "wrong_side_within_line_count": float(wrong_side_within),
        "supported_cross_line_count": float(supported_cross),
        "unsupported_cross_line_count": float(unsupported_cross),
        "wrong_side_cross_line_count": float(wrong_side_cross),
        "boilerplate_line_count": float(boilerplate_lines),
        "aspect_pref_duplicate_count": float(aspect_pref_duplicates),
        "generic_cross_line_count": float(generic_cross_lines),
        "weak_specificity_line_count": float(weak_specificity_lines),
        "explicit_bullet_count": float(explicit_bullets),
    }


class MemIRReward:
    """Evaluate extracted memory by using it to rank candidates."""

    def __init__(
        self,
        llm_name: str = "gpt-4.1-mini",
        prompt_name: str = "llm_rerank",
        prompt_version: str = "v1_mem",
        requires_cross: bool = False,
        return_think: bool = False,
        **kwargs,
    ):
        self.requires_cross = requires_cross
        self.return_think = return_think
        self.margin_weight = float(kwargs.pop("margin_weight", 0.0))
        self.quality_weight = float(kwargs.pop("quality_weight", 0.5))
        self.reward_response_num = int(kwargs.pop("reward_response_num", 5))
        self.rank_retry = kwargs.get("rank_retry", 1)
        self.ranker = RewardRanker(
            llm_name=llm_name,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            return_think=return_think,
            **kwargs,
        )

    def check_format(self, solution: str) -> float:
        has_within = "<within_memory>" in solution and "</within_memory>" in solution
        has_cross = "<cross_memory>" in solution and "</cross_memory>" in solution
        if self.requires_cross:
            return 1.0 if (has_within and has_cross) else 0.0
        return 1.0 if has_within else 0.0

    def _failure_result(self, R_base: float, format_score: float, quality_details: Dict[str, float] = None):
        quality_details = {**default_quality_details(), **(quality_details or {})}
        output = {
            "score": -1.0,
            "rerank_improvement": -R_base,
            "format_score": format_score,
            "rerank_score": 0.0,
            "max_wrong_freq": 0.0,
            "rerank_margin": 0.0,
            "margin_weight": self.margin_weight,
            "quality_weight": self.quality_weight,
            "quality_score": quality_details.get("quality_score", 0.0),
            "margin_reward": 0.0,
            "R_base": R_base,
            "sample_num": self.reward_response_num,
            "time": 0.0,
            "rerank_score_majority_voting": False,
            "ranking_status": False,
        }
        output.update(quality_details)
        return output

    def __call__(self, data_source: Dict[str, Any], solution: str) -> Dict[str, float]:
        R_base = data_source.get("R_base", 0.0)
        format_score = self.check_format(solution)
        if format_score == 0.0:
            return self._failure_result(R_base, 0.0)

        query = data_source["query"]
        candidates = data_source["candidates"]
        metadata = data_source["metadata"]
        pos_product = data_source["pos_product"]
        category_name = data_source.get("category_name", "Unknown")

        documents = []
        for pid in candidates:
            meta = metadata.get(pid, "No metadata")
            if not isinstance(meta, str):
                meta = str(meta)
            documents.append((pid, meta))

        prompt_kwargs = self._prepare_prompt_kwargs(solution, category_name)
        quality_details = score_memory_quality(solution, data_source, requires_cross=self.requires_cross)
        quality_score = quality_details["quality_score"]

        # Same-category memory is required. Cross memory is optional because
        # cross-category history may not safely transfer to the target category.
        if not prompt_kwargs.get("within_memory"):
            return self._failure_result(R_base, format_score, quality_details)

        margin_reward = 0.0
        rerank_score = 0.0
        max_wrong_freq = 0.0
        rerank_margin = 0.0
        result = []
        thinkings = []
        most_common_doc = None
        success = False
        dt = 0.0
        try:
            t0 = time.time()
            most_common_doc, result, success, thinkings = self.ranker.rank(
                query=query,
                documents=documents,
                response_num=self.reward_response_num,
                **prompt_kwargs,
            )
            dt = time.time() - t0
            if not success:
                logging.warning(f"Ranking failed for query: {query}")
            margin_reward, rerank_score, max_wrong_freq, rerank_margin = self._compute_margin_reward(
                result, pos_product
            )
        except Exception as e:
            logging.error(f"Error in reward calculation: {e}")

        final_reward = round(rerank_score + self.quality_weight * quality_score, 4) if result else rerank_score
        output = {
            "score": final_reward,
            "rerank_improvement": final_reward - R_base,
            "format_score": format_score,
            "rerank_score": rerank_score,
            "max_wrong_freq": max_wrong_freq,
            "rerank_margin": rerank_margin,
            "margin_weight": self.margin_weight,
            "quality_weight": self.quality_weight,
            "quality_score": quality_score,
            "margin_reward": margin_reward,
            "R_base": R_base,
            "sample_num": len(result),
            "time": dt,
            "rerank_score_majority_voting": most_common_doc == pos_product,
            "ranking_status": success,
        }
        output.update(quality_details)
        if self.return_think:
            output["thinkings"] = [
                (think, 1.0 if doc_id == pos_product else 0.0) for doc_id, think in zip(result, thinkings)
            ]
        return output

    def _compute_margin_reward(self, result: List[str], pos_product: str) -> Tuple[float, float, float, float]:
        if not result:
            return 0.0, 0.0, 0.0, 0.0
        denominator = self.reward_response_num if self.rank_retry == 1 else len(result)
        if denominator <= 0:
            return 0.0, 0.0, 0.0, 0.0
        counts = Counter(result)
        pos_freq = counts.get(pos_product, 0) / float(denominator)
        wrong_counts = [count for doc_id, count in counts.items() if doc_id != pos_product]
        max_wrong_freq = (max(wrong_counts) / float(denominator)) if wrong_counts else 0.0
        margin = pos_freq - max_wrong_freq
        # The first returned value keeps the historical "margin_reward" logging
        # key, but v2 intentionally uses plain pos_freq as the utility reward.
        return round(pos_freq, 4), round(pos_freq, 4), round(max_wrong_freq, 4), round(margin, 4)

    def _prepare_prompt_kwargs(self, solution: str, category_name: str) -> Dict[str, str]:
        within_mem = strip_evidence_fields(_extract_block(solution, "within_memory"))
        cross_mem = strip_evidence_fields(_extract_block(solution, "cross_memory")) if self.requires_cross else ""
        return {
            "category_name": category_name,
            "within_memory": within_mem,
            "cross_memory": cross_mem,
        }


class RewardW(MemIRReward):
    """Reward function for M_W extraction."""

    def __init__(self, **kwargs):
        super().__init__(requires_cross=False, **kwargs)


class RewardWC(MemIRReward):
    """Reward function for M_WC extraction."""

    def __init__(self, **kwargs):
        super().__init__(requires_cross=True, **kwargs)


_reward_w = None
_reward_wc = None


def compute_score_w(data_source: str, solution_str: str, ground_truth=None, extra_info=None, **kwargs) -> Dict[str, float]:
    global _reward_w
    if _reward_w is None:
        _reward_w = RewardW(**kwargs)
    reward_data = extra_info.get("reward_data", {}) if extra_info else {}
    return _reward_w(reward_data, solution_str)


def compute_score_wc(data_source: str, solution_str: str, ground_truth=None, extra_info=None, **kwargs) -> Dict[str, float]:
    global _reward_wc
    if _reward_wc is None:
        _reward_wc = RewardWC(**kwargs)
    reward_data = extra_info.get("reward_data", {}) if extra_info else {}
    return _reward_wc(reward_data, solution_str)
