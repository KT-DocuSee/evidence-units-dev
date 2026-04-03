"""
OmniDocBench 기반 Retrieval 평가 (통합 버전)

[QA 생성 전략]
  - QA 포함 기준: Strict 범위 충족 시 포함 (엄격한 기준으로 QA 세트 고정)
  - 각 QA에 두 가지 evidence_context 저장:
      evidence_context_strict: 캡션 + 직접 연결 요소 (±3 이내)
      evidence_context_fair  : 캡션 + 주변 관련 텍스트 포함 (±4 이내)
  - qa_id: (image_name + evidence_source + question) SHA-256 해시 → 내용 기반, 안정적

[평가 방식]
  score = LCS길이 / len(gt_context). 분모가 클수록 점수가 낮아짐.

  - Strict 평가: retrieved text vs evidence_context_strict (짧은 gt)
      → 분모가 작아서 w/o EU도 점수가 올라감 → EU delta 보수적 추정
  - Fair 평가  : retrieved text vs evidence_context_fair (긴 gt)
      → 분모가 커서 w/o EU 점수가 내려감 → EU delta 관대한 추정

[비교 실험]
  GT(OmniDocBench)  w/o EU  — baseline
  GT                w/ EU   — EU 효과 상한 (GT bbox 사용)
  Docling           w/o EU  — 파서 독립: Docling 블록 그대로
  Docling           w/ EU   — 파서 독립: Docling + EU
  MinerU            w/o EU  — 파서 독립: MinerU 블록 그대로
  MinerU            w/ EU   — 파서 독립: MinerU + EU

[사용법]
  # QA 생성 + GT 평가
  python eval_retrieval_combined.py \\
    --gt datasets/omnidocbench/source_hf/OmniDocBench.json \\
    --output output/retrieval_eval

  # Docling / MinerU EU 디렉토리 추가 (eu_from_parser.py로 미리 생성)
  python eval_retrieval_combined.py \\
    --gt datasets/omnidocbench/source_hf/OmniDocBench.json \\
    --output output/retrieval_eval \\
    --qas output/retrieval_eval/qas.json \\
    --docling-eu-dir output/eu_docling \\
    --mineru-eu-dir  output/eu_mineru
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ════════════════════════════════════════════════════════════════════
# 범위 파라미터 정의
# ════════════════════════════════════════════════════════════════════

STRICT = dict(visual=3, text=2, footnote=4, title=3)
FAIR   = dict(visual=4, text=4, footnote=5, title=4)


# ════════════════════════════════════════════════════════════════════
# 1. QA 데이터 구조
# ════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalQA:
    qa_id: str
    image_name: str
    question: str
    evidence_context_strict: str   # 좁은 정답 context
    evidence_context_fair: str     # 넓은 정답 context
    evidence_source: str           # table / figure / text


def _make_qa_id(image_name: str, evidence_source: str, question: str) -> str:
    """내용 기반 안정 ID: SHA-256 앞 12자리"""
    key = f"{image_name}:{evidence_source}:{question}"
    return "qa_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _build_context(
    det_idx: int,
    det: dict,
    dets_sorted: list,
    source_type: str,   # "table" | "figure" | "text"
    params: dict,
) -> Tuple[List[str], List[int]]:
    """
    캡션/제목(det)을 기준으로 주변 요소를 수집해 context_parts 반환.
    source_type에 따라 다른 수집 로직 적용.
    """
    caption = det.get("text", "").strip()
    det_order = det.get("order") if det.get("order") is not None else 999
    context_parts = [caption]
    evidence_indices = [det_idx]

    for j, other in enumerate(dets_sorted):
        if j == det_idx:
            continue
        other_order = other.get("order") if other.get("order") is not None else 999
        odiff = abs(other_order - det_order)
        cat = other["category_type"]
        text = other.get("text", "").strip()

        if source_type == "table":
            if cat == "table" and odiff <= params["visual"]:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)
            elif cat == "table_footnote" and odiff <= params["footnote"]:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)
            elif cat == "text_block" and odiff <= params["text"]:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)
            elif cat == "title" and 0 < (det_order - other_order) <= 2:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)

        elif source_type == "figure":
            if cat in ("figure", "figure_footnote") and odiff <= params["visual"]:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)
            elif cat == "text_block" and odiff <= params["text"]:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)
            elif cat == "title" and 0 < (det_order - other_order) <= 2:
                if text:
                    context_parts.append(text)
                evidence_indices.append(j)

        elif source_type == "text":
            if cat == "text_block":
                order_diff = other_order - det_order
                if 0 < order_diff <= params["title"]:
                    if text:
                        context_parts.append(text)
                        evidence_indices.append(j)

    return context_parts, evidence_indices


# ════════════════════════════════════════════════════════════════════
# 2. QA 생성: Strict 기준으로 선정, Strict+Fair context 동시 저장
# ════════════════════════════════════════════════════════════════════

def _is_english(page_entry: dict) -> bool:
    lang = page_entry.get("page_info", {}).get("page_attribute", {}).get("language", "")
    return lang == "english"


def build_shared_qas(gt_path: str, limit: int = 0, english_only: bool = False) -> List[RetrievalQA]:
    """
    [QA 포함 기준]
      table_caption → Strict context_parts >= 2 일 때 포함
      figure_caption → 항상 포함 (Strict와 동일)
      title          → Strict context_parts >= 2 일 때 포함

    [두 context 생성]
      evidence_context_strict: STRICT 파라미터로 수집한 context
      evidence_context_fair  : FAIR   파라미터로 수집한 context

    english_only: page_attribute.language == "english" 인 페이지만 사용
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    if limit > 0:
        pages = pages[:limit]
    if english_only:
        pages = [p for p in pages if _is_english(p)]
        logger.info(f"  english_only: {len(pages)} pages")

    qas: List[RetrievalQA] = []
    seen_ids: set = set()  # 중복 qa_id 방지

    for page_entry in pages:
        page_info = page_entry.get("page_info", {})
        image_name = Path(page_info.get("image_path", "unknown")).stem
        dets = page_entry.get("layout_dets", [])
        dets_sorted = sorted(dets, key=lambda d: d.get("order") if d.get("order") is not None else 999)

        # ── table_caption ──
        for i, det in enumerate(dets_sorted):
            if det["category_type"] != "table_caption":
                continue
            caption = det.get("text", "").strip()
            if not caption:
                continue

            strict_parts, _ = _build_context(i, det, dets_sorted, "table", STRICT)
            if len(strict_parts) < 2:
                continue  # Strict 기준 미충족 → QA 제외

            fair_parts, _ = _build_context(i, det, dets_sorted, "table", FAIR)

            qa_id = _make_qa_id(image_name, "table", caption)
            if qa_id in seen_ids:
                continue
            seen_ids.add(qa_id)

            qas.append(RetrievalQA(
                qa_id=qa_id,
                image_name=image_name,
                question=caption,
                evidence_context_strict="\n".join(strict_parts),
                evidence_context_fair="\n".join(fair_parts),
                evidence_source="table",
            ))

        # ── figure_caption ──
        for i, det in enumerate(dets_sorted):
            if det["category_type"] != "figure_caption":
                continue
            caption = det.get("text", "").strip()
            if not caption:
                continue

            strict_parts, _ = _build_context(i, det, dets_sorted, "figure", STRICT)
            fair_parts, _   = _build_context(i, det, dets_sorted, "figure", FAIR)

            qa_id = _make_qa_id(image_name, "figure", caption)
            if qa_id in seen_ids:
                continue
            seen_ids.add(qa_id)

            qas.append(RetrievalQA(
                qa_id=qa_id,
                image_name=image_name,
                question=caption,
                evidence_context_strict="\n".join(strict_parts),
                evidence_context_fair="\n".join(fair_parts),
                evidence_source="figure",
            ))

        # ── title ──
        for i, det in enumerate(dets_sorted):
            if det["category_type"] != "title":
                continue
            title_text = det.get("text", "").strip()
            if len(title_text) < 5:
                continue

            strict_parts, _ = _build_context(i, det, dets_sorted, "text", STRICT)
            if len(strict_parts) < 2:
                continue  # Strict 기준 미충족 → QA 제외

            fair_parts, _ = _build_context(i, det, dets_sorted, "text", FAIR)

            qa_id = _make_qa_id(image_name, "text", title_text)
            if qa_id in seen_ids:
                continue
            seen_ids.add(qa_id)

            qas.append(RetrievalQA(
                qa_id=qa_id,
                image_name=image_name,
                question=title_text,
                evidence_context_strict="\n".join(strict_parts),
                evidence_context_fair="\n".join(fair_parts),
                evidence_source="text",
            ))

    logger.info(f"Generated {len(qas)} shared QA pairs from {len(pages)} pages")
    source_dist = defaultdict(int)
    for qa in qas:
        source_dist[qa.evidence_source] += 1
    logger.info(f"  Source distribution: {dict(source_dist)}")
    return qas


def load_qas_from_file(qas_path: str) -> List[RetrievalQA]:
    """저장된 qas.json 로드 (qa_id 고정 재사용)"""
    with open(qas_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qas = []
    for d in data:
        qas.append(RetrievalQA(
            qa_id=d["qa_id"],
            image_name=d["image_name"],
            question=d["question"],
            evidence_context_strict=d["evidence_context_strict"],
            evidence_context_fair=d["evidence_context_fair"],
            evidence_source=d["evidence_source"],
        ))
    logger.info(f"Loaded {len(qas)} QA pairs from {qas_path}")
    return qas


def save_qas(qas: List[RetrievalQA], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{
            "qa_id": qa.qa_id,
            "image_name": qa.image_name,
            "question": qa.question,
            "evidence_context_strict": qa.evidence_context_strict,
            "evidence_context_fair": qa.evidence_context_fair,
            "evidence_source": qa.evidence_source,
        } for qa in qas], f, ensure_ascii=False, indent=2)
    logger.info(f"QAs saved: {path}")


# ════════════════════════════════════════════════════════════════════
# 3. Chunk 생성
# ════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    chunk_id: str
    image_name: str
    text: str
    source: str


def build_chunks_without_eu(gt_path: str, limit: int = 0, english_only: bool = False) -> Dict[str, List[Chunk]]:
    """w/o EU: GT 개별 요소를 그대로 chunk"""
    with open(gt_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    if limit > 0:
        pages = pages[:limit]
    if english_only:
        pages = [p for p in pages if _is_english(p)]

    skip_types = {"abandon", "need_mask", "table_mask", "text_mask", "page_number", "header", "footer"}
    chunks_by_page = {}

    for page_entry in pages:
        image_name = Path(page_entry.get("page_info", {}).get("image_path", "unknown")).stem
        dets = page_entry.get("layout_dets", [])
        chunks = []
        for i, det in enumerate(sorted(dets, key=lambda d: d.get("order") if d.get("order") is not None else 999)):
            if det["category_type"] in skip_types or det.get("ignore", False):
                continue
            text = det.get("text", "").strip()
            if text:
                chunks.append(Chunk(chunk_id=f"{image_name}_{i}", image_name=image_name, text=text, source="without_eu"))
        if chunks:
            chunks_by_page[image_name] = chunks

    return chunks_by_page


def build_chunks_with_eu(gt_path: str, text_model, limit: int = 0, english_only: bool = False) -> Dict[str, List[Chunk]]:
    """w/ EU: EU 단위 chunk"""
    from eu_from_parser import omnidocbench_to_canon_nodes, run_eu_pipeline

    with open(gt_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    if limit > 0:
        pages = pages[:limit]
    if english_only:
        pages = [p for p in pages if _is_english(p)]

    chunks_by_page = {}

    for page_entry in pages:
        image_name = Path(page_entry.get("page_info", {}).get("image_path", "unknown")).stem
        canon_nodes = omnidocbench_to_canon_nodes(page_entry, image_name, text_model)
        if not canon_nodes:
            continue

        eus = run_eu_pipeline(canon_nodes)
        nodes_by_id = {n.node_id: n for n in canon_nodes}
        chunks = []
        for eu in eus:
            members = sorted(
                [(nodes_by_id[m].order, nodes_by_id[m].text.strip())
                 for m in eu.member_node_ids
                 if m in nodes_by_id and nodes_by_id[m].text.strip()],
                key=lambda x: x[0],
            )
            eu_text = "\n".join(t for _, t in members)
            if eu_text.strip():
                chunks.append(Chunk(chunk_id=eu.eu_id, image_name=image_name, text=eu_text, source="with_eu"))
        if chunks:
            chunks_by_page[image_name] = chunks

    return chunks_by_page


# ════════════════════════════════════════════════════════════════════
# 4. Retrieval 평가
# ════════════════════════════════════════════════════════════════════

def lcs_score(retrieved: str, ground_truth: str) -> float:
    """OHR-Bench 동일 LCS 점수 (LCS길이 / GT길이)"""
    if not retrieved or not ground_truth:
        return 0.0
    m, n = len(retrieved), len(ground_truth)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if retrieved[i - 1] == ground_truth[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n] / n if n > 0 else 0.0


def evaluate_retrieval(
    qas: List[RetrievalQA],
    chunks_by_page: Dict[str, List[Chunk]],
    embed_model,
    top_k: int = 3,
    label: str = "",
    context_mode: str = "strict",   # "strict" | "fair"
) -> Tuple[dict, List[dict]]:
    """
    Retrieval 평가.
    context_mode:
      score = LCS길이 / len(gt_context) 이므로 분모(gt_context 길이)가 클수록 점수가 낮아짐.

      "strict" → gt_context가 짧음 (좁은 범위) → 분모 작음
                 → w/o EU도 점수가 올라가서 EU와의 delta가 작게 나옴
                 → EU 개선 효과의 보수적 추정 (w/o EU에게 유리한 조건)

      "fair"   → gt_context가 길음 (EU가 묶는 범위와 동일)→ 분모 큼
                 → w/o EU 점수가 내려가서 EU와의 delta가 크게 나옴
                 → EU 개선 효과의 관대한 추정 (EU 설계 의도에 맞는 조건)

    Returns:
        (result_dict, per_qa_traces)
        per_qa_traces: QA별 retrieved chunks, similarity, LCS 등 상세 기록
    """
    # 페이지별 chunk 임베딩 캐시
    page_embeds: Dict[str, np.ndarray] = {}
    for page_name, chunks in chunks_by_page.items():
        texts = [c.text for c in chunks]
        if texts:
            embs = embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            page_embeds[page_name] = embs

    MAX_SCAN_K = 10
    results_by_source = defaultdict(lambda: {
        "lcs_sum": 0.0, "hit_sum": 0, "count": 0,
        "recall_at": {1: 0, 2: 0, 3: 0, 5: 0},
        "min_k_sum": 0.0, "min_k_count": 0,
        "token_sum": 0, "token_count": 0,
    })
    total_lcs = 0.0
    total_hit = 0
    total_count = 0
    recall_at = {1: 0, 2: 0, 3: 0, 5: 0}
    min_k_sum = 0.0
    min_k_found = 0
    token_when_hit_sum = 0
    token_when_hit_count = 0

    per_qa_traces: List[dict] = []

    for qa in qas:
        chunks = chunks_by_page.get(qa.image_name)
        embs = page_embeds.get(qa.image_name)
        if chunks is None or embs is None:
            # Parser has no output for this page → score 0 (fair fixed-denominator evaluation)
            total_count += 1
            src = qa.evidence_source
            results_by_source[src]["count"] += 1
            per_qa_traces.append({
                "qa_id": qa.qa_id,
                "retrieved_chunks": [],
                "retrieved_text": "",
                "gt_context": (
                    qa.evidence_context_strict if context_mode == "strict"
                    else qa.evidence_context_fair
                ),
                "lcs": 0.0,
                "hit": False,
                "min_k": None,
            })
            continue

        # context 선택
        gt_context = (
            qa.evidence_context_strict if context_mode == "strict"
            else qa.evidence_context_fair
        )

        q_emb = embed_model.encode([qa.question], convert_to_numpy=True)[0]
        sims = np.dot(embs, q_emb) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
        all_ranked = np.argsort(sims)[::-1][:MAX_SCAN_K]

        # top_k LCS
        top_indices = all_ranked[:top_k]
        ret_text = "\n\n".join(chunks[idx].text for idx in top_indices)
        lcs = lcs_score(ret_text, gt_context)
        hit = 1 if lcs > 0.3 else 0

        total_lcs += lcs
        total_hit += hit
        total_count += 1

        src = qa.evidence_source
        results_by_source[src]["lcs_sum"] += lcs
        results_by_source[src]["hit_sum"] += hit
        results_by_source[src]["count"] += 1

        # Recall@K
        found_k = None
        for k in [1, 2, 3, 5]:
            k_text = "\n\n".join(chunks[idx].text for idx in all_ranked[:k])
            if lcs_score(k_text, gt_context) > 0.3:
                recall_at[k] += 1
                results_by_source[src]["recall_at"][k] += 1
                if found_k is None:
                    found_k = k

        # Min-K
        if found_k is not None:
            min_k_sum += found_k
            min_k_found += 1
            results_by_source[src]["min_k_sum"] += found_k
            results_by_source[src]["min_k_count"] += 1

        # 토큰 효율
        top_k_chars = sum(len(chunks[idx].text) for idx in top_indices)
        if hit:
            token_when_hit_sum += top_k_chars
            token_when_hit_count += 1
            results_by_source[src]["token_sum"] += top_k_chars
            results_by_source[src]["token_count"] += 1

        # ── QA별 상세 trace ──
        per_qa_traces.append({
            "qa_id": qa.qa_id,
            "retrieved_chunks": [
                {
                    "rank": rank + 1,
                    "chunk_id": chunks[idx].chunk_id,
                    "sim": round(float(sims[idx]), 4),
                    "text": chunks[idx].text,
                }
                for rank, idx in enumerate(top_indices)
            ],
            "retrieved_text": ret_text,
            "gt_context": gt_context,
            "lcs": round(lcs, 4),
            "hit": bool(hit),
            "min_k": found_k,
        })

    avg_lcs = total_lcs / total_count if total_count > 0 else 0
    hit_rate = total_hit / total_count if total_count > 0 else 0

    result = {
        "label": label,
        "context_mode": context_mode,
        "avg_lcs": round(avg_lcs, 4),
        "hit_rate": round(hit_rate, 4),
        "total": total_count,
        "recall_at": {k: round(v / total_count, 4) if total_count > 0 else 0 for k, v in recall_at.items()},
        "avg_min_k": round(min_k_sum / min_k_found, 2) if min_k_found > 0 else None,
        "avg_token_per_hit": round(token_when_hit_sum / token_when_hit_count) if token_when_hit_count > 0 else None,
        "by_source": {},
    }
    for src, r in results_by_source.items():
        cnt = r["count"]
        result["by_source"][src] = {
            "avg_lcs": round(r["lcs_sum"] / cnt, 4) if cnt > 0 else 0,
            "hit_rate": round(r["hit_sum"] / cnt, 4) if cnt > 0 else 0,
            "count": cnt,
            "recall_at": {k: round(v / cnt, 4) if cnt > 0 else 0 for k, v in r["recall_at"].items()},
            "avg_min_k": round(r["min_k_sum"] / r["min_k_count"], 2) if r["min_k_count"] > 0 else None,
            "avg_token_per_hit": round(r["token_sum"] / r["token_count"]) if r["token_count"] > 0 else None,
        }
    return result, per_qa_traces


# ════════════════════════════════════════════════════════════════════
# 5. 외부 파서 EU 디렉토리에서 Chunk 로드
# ════════════════════════════════════════════════════════════════════

def build_chunks_from_eu_dir(
    eu_dir: str,
    without_eu: bool = False,
    qa_image_names: Optional[set] = None,
) -> Dict[str, List[Chunk]]:
    """
    eu_from_parser.py 출력 디렉토리에서 Chunk 로드.

    eu_dir/
      with_eu/   {image_name}.json   ← eu_to_dict() 포맷
      without_eu/{image_name}.json   ← nodes_to_chunks_without_eu() 포맷

    without_eu=False → with_eu/ 디렉토리 읽음
    without_eu=True  → without_eu/ 디렉토리 읽음

    qa_image_names: QA에 등장하는 image_name 집합. 지정 시 해당 페이지만 로드.
    """
    sub = "without_eu" if without_eu else "with_eu"
    target_dir = Path(eu_dir) / sub
    if not target_dir.exists():
        logger.warning(f"  {target_dir} 가 존재하지 않습니다.")
        return {}

    source_label = "without_eu" if without_eu else "with_eu"
    chunks_by_page: Dict[str, List[Chunk]] = {}

    for json_file in sorted(target_dir.glob("*.json")):
        image_name = json_file.stem
        if qa_image_names is not None and image_name not in qa_image_names:
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            items = json.load(f)

        chunks = []
        if without_eu:
            # nodes_to_chunks_without_eu() 포맷: [{chunk_id, text, canon_role, ...}]
            for item in items:
                text = (item.get("text") or "").strip()
                if text:
                    chunks.append(Chunk(
                        chunk_id=item.get("chunk_id", f"{image_name}_{len(chunks)}"),
                        image_name=image_name,
                        text=text,
                        source=source_label,
                    ))
        else:
            # eu_to_dict() 포맷: [{eu_id, elements: [{text, order}]}]
            for eu_item in items:
                elements = eu_item.get("elements", [])
                texts = sorted(
                    [(e.get("order", 9999), e.get("text", "").strip()) for e in elements],
                    key=lambda x: x[0],
                )
                eu_text = "\n".join(t for _, t in texts if t)
                if eu_text.strip():
                    chunks.append(Chunk(
                        chunk_id=eu_item.get("eu_id", f"{image_name}_{len(chunks)}"),
                        image_name=image_name,
                        text=eu_text,
                        source=source_label,
                    ))

        if chunks:
            chunks_by_page[image_name] = chunks

    return chunks_by_page


# ════════════════════════════════════════════════════════════════════
# 6. 결과 출력
# ════════════════════════════════════════════════════════════════════

def report_parser_coverage(
    chunks_a: Dict[str, List["Chunk"]],
    chunks_b: Dict[str, List["Chunk"]],
    all_qas: list,
    label_a: str = "MinerU",
    label_b: str = "Docling",
) -> Tuple[set, list]:
    """
    두 파서의 EU 출력 파일을 파일명(image_name) 기준으로 비교한다.
    - 공통 페이지만 cross-parser 평가에 사용
    - 빠진 파일 수와 영향받는 QA 수를 출력한다
    Returns: (common_image_names, filtered_qas_for_common_pages)
    """
    pages_a = set(chunks_a.keys())
    pages_b = set(chunks_b.keys())
    common  = pages_a & pages_b
    only_a  = pages_a - pages_b
    only_b  = pages_b - pages_a

    qa_only_a   = [qa for qa in all_qas if qa.image_name in only_a]
    qa_only_b   = [qa for qa in all_qas if qa.image_name in only_b]
    common_qas  = [qa for qa in all_qas if qa.image_name in common]

    sep = "=" * 72
    print(f"\n{sep}")
    print("PARSER COVERAGE REPORT  (cross-parser comparison)")
    print(sep)
    print(f"  {label_a:<20s}: {len(pages_a):5d} pages with EU output")
    print(f"  {label_b:<20s}: {len(pages_b):5d} pages with EU output")
    print(f"  Common (evaluated)  : {len(common):5d} pages")
    print(f"  Missing from {label_b:<10s}: {len(only_a):5d} pages  "
          f"→ {len(qa_only_a)} QAs excluded from cross-parser eval")
    print(f"  Missing from {label_a:<10s}: {len(only_b):5d} pages  "
          f"→ {len(qa_only_b)} QAs excluded from cross-parser eval")
    print(f"  Cross-parser QA set : {len(common_qas):5d} QAs  "
          f"(out of {len(all_qas)} total)")
    print(sep)

    if only_a:
        logger.info(f"Pages in {label_a} but missing from {label_b} ({len(only_a)}): "
                    + ", ".join(sorted(only_a)[:10])
                    + (" ..." if len(only_a) > 10 else ""))
    if only_b:
        logger.info(f"Pages in {label_b} but missing from {label_a} ({len(only_b)}): "
                    + ", ".join(sorted(only_b)[:10])
                    + (" ..." if len(only_b) > 10 else ""))

    return common, common_qas


def print_comparison(results: List[dict]):
    print("\n" + "=" * 100)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 100)

    # ── LCS + Hit Rate ──
    print(f"\n{'Method':<30s} {'Mode':<8s} {'Avg LCS':>8s} {'Hit@3':>8s} {'N':>6s}")
    print("-" * 65)
    for r in results:
        print(f"{r['label']:<30s} {r['context_mode']:<8s} {r['avg_lcs']:>8.4f} {r['hit_rate']:>8.4f} {r['total']:>6d}")

    # ── Recall@K + 토큰 효율 ──
    print(f"\n{'Method':<30s} {'Mode':<8s} {'R@1':>7s} {'R@2':>7s} {'R@3':>7s} {'R@5':>7s} {'MinK':>7s} {'AvgTok':>8s}")
    print("-" * 85)
    for r in results:
        ra = r.get("recall_at", {})
        mk = r.get("avg_min_k")
        tk = r.get("avg_token_per_hit")
        print(f"{r['label']:<30s} {r['context_mode']:<8s} "
              f"{ra.get(1,0):>7.4f} {ra.get(2,0):>7.4f} {ra.get(3,0):>7.4f} {ra.get(5,0):>7.4f} "
              f"{mk if mk else 'N/A':>7} {tk if tk else 'N/A':>8}")

    # ── source별 ──
    all_sources = sorted({src for r in results for src in r.get("by_source", {})})
    if all_sources:
        print("\n" + "-" * 100)
        print("BY EVIDENCE SOURCE:")
        for src in all_sources:
            print(f"\n  [{src}]")
            print(f"  {'Method':<28s} {'Mode':<8s} {'LCS':>7s} {'R@1':>7s} {'R@3':>7s} {'MinK':>6s} {'N':>5s}")
            for r in results:
                s = r.get("by_source", {}).get(src)
                if s:
                    ra = s.get("recall_at", {})
                    mk = s.get("avg_min_k")
                    print(f"  {r['label']:<28s} {r['context_mode']:<8s} "
                          f"{s['avg_lcs']:>7.4f} {ra.get(1,0):>7.4f} {ra.get(3,0):>7.4f} "
                          f"{mk if mk else 'N/A':>6} {s['count']:>5d}")

    # ── Delta vs baseline (GT w/o EU) ──
    # 같은 context_mode끼리 baseline(GT w/o EU)과 비교
    if len(results) >= 2:
        print("\n" + "-" * 80)
        print("DELTA (vs GT w/o EU, same context_mode):")
        baselines = {r["context_mode"]: r for r in results if r["label"] == "GT w/o EU"}
        for r in results:
            base = baselines.get(r["context_mode"])
            if base is None or r["label"] == "GT w/o EU":
                continue
            diff_lcs = r["avg_lcs"] - base["avg_lcs"]
            diff_hit = r["hit_rate"] - base["hit_rate"]
            diff_r1 = r["recall_at"].get(1, 0) - base["recall_at"].get(1, 0)
            print(f"  [{r['context_mode']}] {r['label']:<30s}: "
                  f"LCS={diff_lcs:+.4f}  HitRate={diff_hit:+.4f}  R@1={diff_r1:+.4f}")

    print("=" * 100)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="OmniDocBench Retrieval Eval (Combined Strict+Fair)")
    parser.add_argument("--gt",     required=True, help="OmniDocBench GT JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--qas",    default="",    help="기존 qas.json 경로 (지정 시 QA 재생성 생략)")
    parser.add_argument("--limit",          type=int,  default=0,  help="Max pages (0=all)")
    parser.add_argument("--top-k",          type=int,  default=3,  help="Retrieval top-K")
    parser.add_argument("--english-only",   action="store_true",   help="language==english 페이지만 사용")
    parser.add_argument("--docling-eu-dir", default="",            help="eu_from_parser --source docling 출력 디렉토리")
    parser.add_argument("--mineru-eu-dir",  default="",            help="eu_from_parser --source mineru 출력 디렉토리")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ──
    import torch
    from sentence_transformers import SentenceTransformer
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    logger.info("Loading text/embed model...")
    from eu_from_parser import Config
    text_model = SentenceTransformer(Config.ST_STRUCTURE_PATH)
    text_model.load_state_dict(torch.load(Config.TEXT_EMBED_PATH, map_location="cpu"))
    text_model.eval()
    embed_model = text_model
    logger.info("Models loaded.")

    # ── Step 1: QA 로드 또는 생성 ──
    qa_path = Path(args.qas) if args.qas else output_dir / "qas.json"
    en = args.english_only
    if en:
        logger.info("english_only mode: language==english 페이지만 사용")

    if args.qas and Path(args.qas).exists():
        logger.info(f"Loading existing QAs from {args.qas}")
        qas = load_qas_from_file(args.qas)
    else:
        logger.info("Generating shared QAs (strict criteria, both contexts)...")
        qas = build_shared_qas(args.gt, args.limit, english_only=en)
        save_qas(qas, output_dir / "qas.json")

    all_results = []

    # trace 병합용
    qa_meta = {qa.qa_id: {
        "qa_id": qa.qa_id,
        "image_name": qa.image_name,
        "question": qa.question,
        "evidence_source": qa.evidence_source,
        "gt_context_strict": qa.evidence_context_strict,
        "gt_context_fair": qa.evidence_context_fair,
        "gt_without_eu": {},
        "gt_with_eu": {},
        "docling_without_eu": {},
        "docling_with_eu": {},
        "mineru_without_eu": {},
        "mineru_with_eu": {},
    } for qa in qas}

    # QA에 등장하는 image_name 집합 (파서 EU 로드 시 필터링용)
    qa_image_names = {qa.image_name for qa in qas}

    # ── Step 2: GT w/o EU ──
    logger.info("Building chunks: GT w/o EU...")
    wo_chunks = build_chunks_without_eu(args.gt, args.limit, english_only=en)
    logger.info(f"  {len(wo_chunks)} pages, {sum(len(v) for v in wo_chunks.values())} chunks")

    def _collect_traces(traces, qa_meta_key: str):
        for t in traces:
            if t["qa_id"] in qa_meta:
                qa_meta[t["qa_id"]][qa_meta_key][t.get("_mode", "strict")] = {
                    "retrieved_chunks": t["retrieved_chunks"],
                    "lcs": t["lcs"],
                    "hit": t["hit"],
                    "min_k": t["min_k"],
                }

    for mode in ["strict", "fair"]:
        r, traces = evaluate_retrieval(qas, wo_chunks, embed_model, args.top_k, "GT w/o EU", context_mode=mode)
        all_results.append(r)
        logger.info(f"  [GT w/o EU | {mode}] LCS={r['avg_lcs']:.4f}, HitRate={r['hit_rate']:.4f}")
        for t in traces:
            t["_mode"] = mode
        _collect_traces(traces, "gt_without_eu")

    # ── Step 3: GT w/ EU ──
    logger.info("Building chunks: GT w/ EU...")
    w_chunks = build_chunks_with_eu(args.gt, text_model, args.limit, english_only=en)
    logger.info(f"  {len(w_chunks)} pages, {sum(len(v) for v in w_chunks.values())} chunks")

    for mode in ["strict", "fair"]:
        r, traces = evaluate_retrieval(qas, w_chunks, embed_model, args.top_k, "GT w/ EU", context_mode=mode)
        all_results.append(r)
        logger.info(f"  [GT w/ EU | {mode}] LCS={r['avg_lcs']:.4f}, HitRate={r['hit_rate']:.4f}")
        for t in traces:
            t["_mode"] = mode
        _collect_traces(traces, "gt_with_eu")

    # ── Step 4: Docling w/o EU + w/ EU (선택) ──
    if args.docling_eu_dir:
        logger.info(f"Building chunks: Docling w/o EU ({args.docling_eu_dir})...")
        docling_wo = build_chunks_from_eu_dir(args.docling_eu_dir, without_eu=True, qa_image_names=qa_image_names)
        logger.info(f"  {len(docling_wo)} pages, {sum(len(v) for v in docling_wo.values())} chunks")
        for mode in ["strict", "fair"]:
            r, traces = evaluate_retrieval(qas, docling_wo, embed_model, args.top_k, "Docling w/o EU", context_mode=mode)
            all_results.append(r)
            logger.info(f"  [Docling w/o EU | {mode}] LCS={r['avg_lcs']:.4f}, HitRate={r['hit_rate']:.4f}")
            for t in traces:
                t["_mode"] = mode
            _collect_traces(traces, "docling_without_eu")

        logger.info(f"Building chunks: Docling w/ EU ({args.docling_eu_dir})...")
        docling_w = build_chunks_from_eu_dir(args.docling_eu_dir, without_eu=False, qa_image_names=qa_image_names)
        logger.info(f"  {len(docling_w)} pages, {sum(len(v) for v in docling_w.values())} chunks")
        for mode in ["strict", "fair"]:
            r, traces = evaluate_retrieval(qas, docling_w, embed_model, args.top_k, "Docling w/ EU", context_mode=mode)
            all_results.append(r)
            logger.info(f"  [Docling w/ EU | {mode}] LCS={r['avg_lcs']:.4f}, HitRate={r['hit_rate']:.4f}")
            for t in traces:
                t["_mode"] = mode
            _collect_traces(traces, "docling_with_eu")

    # ── Step 5: MinerU w/o EU + w/ EU (선택) ──
    if args.mineru_eu_dir:
        logger.info(f"Building chunks: MinerU w/o EU ({args.mineru_eu_dir})...")
        mineru_wo = build_chunks_from_eu_dir(args.mineru_eu_dir, without_eu=True, qa_image_names=qa_image_names)
        logger.info(f"  {len(mineru_wo)} pages, {sum(len(v) for v in mineru_wo.values())} chunks")
        for mode in ["strict", "fair"]:
            r, traces = evaluate_retrieval(qas, mineru_wo, embed_model, args.top_k, "MinerU w/o EU", context_mode=mode)
            all_results.append(r)
            logger.info(f"  [MinerU w/o EU | {mode}] LCS={r['avg_lcs']:.4f}, HitRate={r['hit_rate']:.4f}")
            for t in traces:
                t["_mode"] = mode
            _collect_traces(traces, "mineru_without_eu")

        logger.info(f"Building chunks: MinerU w/ EU ({args.mineru_eu_dir})...")
        mineru_w = build_chunks_from_eu_dir(args.mineru_eu_dir, without_eu=False, qa_image_names=qa_image_names)
        logger.info(f"  {len(mineru_w)} pages, {sum(len(v) for v in mineru_w.values())} chunks")
        for mode in ["strict", "fair"]:
            r, traces = evaluate_retrieval(qas, mineru_w, embed_model, args.top_k, "MinerU w/ EU", context_mode=mode)
            all_results.append(r)
            logger.info(f"  [MinerU w/ EU | {mode}] LCS={r['avg_lcs']:.4f}, HitRate={r['hit_rate']:.4f}")
            for t in traces:
                t["_mode"] = mode
            _collect_traces(traces, "mineru_with_eu")

    # ── Cross-parser comparison on common pages only ──
    if args.docling_eu_dir and args.mineru_eu_dir:
        # use with_eu dicts (already loaded above); without_eu follows same page set
        common_pages, common_qas = report_parser_coverage(
            mineru_w, docling_w, qas, label_a="MinerU", label_b="Docling"
        )
        if common_qas:
            logger.info(f"Re-evaluating cross-parser methods on {len(common_qas)} common-page QAs...")
            cross_results = []
            for label, chunks_dict, trace_key in [
                ("GT w/o EU [common]",      wo_chunks,  None),
                ("GT w/ EU [common]",       w_chunks,   None),
                ("MinerU w/o EU [common]",  mineru_wo,  None),
                ("MinerU w/ EU [common]",   mineru_w,   None),
                ("Docling w/o EU [common]", docling_wo, None),
                ("Docling w/ EU [common]",  docling_w,  None),
            ]:
                for mode in ["strict", "fair"]:
                    r, _ = evaluate_retrieval(
                        common_qas, chunks_dict, embed_model,
                        args.top_k, label, context_mode=mode
                    )
                    cross_results.append(r)

            print("\n\n" + "=" * 72)
            print("CROSS-PARSER RESULTS  (common pages only)")
            print("=" * 72)
            print_comparison(cross_results)

            cross_path = output_dir / "retrieval_results_crossparser.json"
            with open(cross_path, "w", encoding="utf-8") as f:
                json.dump(cross_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Cross-parser results saved: {cross_path}")

    # ── 출력 및 저장 ──
    print_comparison(all_results)

    result_path = output_dir / "retrieval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved: {result_path}")

    # ── QA별 상세 trace 저장 ──
    # 구조: QA 1개당 question / gt_context(strict,fair) / w/o EU 결과 / w/ EU 결과
    trace_path = output_dir / "retrieval_trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(list(qa_meta.values()), f, ensure_ascii=False, indent=2)
    logger.info(f"Trace saved: {trace_path}")


if __name__ == "__main__":
    main()
