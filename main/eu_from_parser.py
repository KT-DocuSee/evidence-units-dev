"""
PaddleOCR-VL 1.5 / OmniDocBench GT / Docling / MinerU → Canonical Role 정규화 → EU 생성 파이프라인

Neo4j 없이 독립 실행 가능. main.py의 EU 알고리즘을 그대로 사용.

사용법:
  # OmniDocBench GT로 EU 생성
  python eu_from_parser.py --source omnidocbench \
      --gt datasets/omnidocbench/source_hf/OmniDocBench.json \
      --images datasets/omnidocbench/source_hf/images \
      --output output/eu_omnidocbench

  # PaddleOCR-VL raw json으로 EU 생성
  python eu_from_parser.py --source paddleocr_vl \
      --input-dir /path/to/paddleocr_vl_json_raw/ \
      --images /path/to/images/ \
      --output /path/to/output/

  # Docling으로 OmniDocBench 영어 이미지 파싱 + EU 생성
  python eu_from_parser.py --source docling \
      --images datasets/omnidocbench/source_hf/images \
      --output output/eu_docling \
      --limit 10

  # MinerU(magic-pdf)로 OmniDocBench 영어 이미지 파싱 + EU 생성
  # 사전 모델 다운로드: mineru-models-download (또는 magic-pdf-models-download)
  python eu_from_parser.py --source mineru \
      --images datasets/omnidocbench/source_hf/images \
      --output output/eu_mineru \
      --limit 10
"""
from __future__ import annotations

import os
import re
import sys
import json
import uuid
import hashlib
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# ── main.py 로직 임포트 (같은 디렉토리) ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docusee.KT_DocuSee_Organizer.optimized_matching import cosine_sim

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ════════════════════════════════════════════════════════════════════
# Config (main.py 와 동일)
# ════════════════════════════════════════════════════════════════════
class Config:
    ST_STRUCTURE_PATH = "/Users/aleydis/Documents/dev/standardized-evidence-units/docusee/model/ontology/ko-sbert"
    TEXT_EMBED_PATH = "/Users/aleydis/Documents/dev/standardized-evidence-units/docusee/model/ontology/model.pt"

    TEXT_CLUSTER_GAP_Y = 0.07
    TEXT_CLUSTER_ORDER_GAP = 3
    TEXT_CLUSTER_X_ALIGN_TAU = 0.18
    HEADER_ATTACH_MAX_ORDER_GAP = 8


# ════════════════════════════════════════════════════════════════════
# CanonNode / MaterializedEU (main.py 와 동일)
# ════════════════════════════════════════════════════════════════════
@dataclass
class CanonNode:
    node_id: str
    observed_type: str
    canon_role: str
    text: str
    bbox: Tuple[float, float, float, float]  # normalized 0~1
    page: str
    doc: str
    order: int
    parser_score: float = 0.0
    text_emb: Optional[List[float]] = None
    vision_emb: Optional[List[float]] = None


@dataclass
class MaterializedEU:
    eu_id: str
    kind: str
    page: str
    primary_id: str
    member_node_ids: List[str] = field(default_factory=list)
    edge_ids: List[Tuple[str, str, str]] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
# 특허 핵심 1: 정규 역할(Canonical Role) 정규화
# ════════════════════════════════════════════════════════════════════

# ── Step 1: 파서별 TYPE_MAP (알려진 파서용 직접 매핑) ──
OMNIDOCBENCH_TYPE_MAP = {
    "title": "section_header",
    "text_block": "support_paragraph",
    "table": "table",
    "table_caption": "support_paragraph",
    "table_footnote": "support_paragraph",
    "figure": "picture",
    "figure_caption": "support_paragraph",
    "figure_footnote": "support_paragraph",
    "chart": "chart",
    "header": "page_header",
    "footer": "page_header",
    "page_number": "page_header",
    "page_footnote": "support_paragraph",
    "reference": "support_paragraph",
    "equation_isolated": "support_paragraph",
    "equation_caption": "support_paragraph",
    "equation_explanation": "support_paragraph",
    "code_txt": "support_paragraph",
    "code_txt_caption": "support_paragraph",
    "abandon": "plain_text",
    "need_mask": "plain_text",
    "table_mask": "plain_text",
    "text_mask": "plain_text",
}

PADDLEOCR_VL_TYPE_MAP = {
    "title": "section_header",
    "text": "support_paragraph",
    "paragraph": "support_paragraph",
    "table": "table",
    "figure": "picture",
    "image": "picture",
    "chart": "chart",
    "header": "page_header",
    "footer": "page_header",
    "caption": "support_paragraph",
    "formula": "support_paragraph",
    "equation": "support_paragraph",
    "seal": "plain_text",
    "stamp": "plain_text",
    "unknown": "plain_text",
}

# ── Step 2: 텍스트 패턴 매칭 (파서 무관, 우선 적용) ──
BRACKET_TITLE_RE = re.compile(r"^\[\s*.+\s*\]$")
UNIT_RE = re.compile(r"^\(\s*[단Uu][위nN][이iI]?[tT]?\s*:")


def assign_canon_role(raw_label: str, text: str, type_map: Dict[str, str]) -> str:
    """
    특허 정규화 3단계:
      Step 1: 텍스트 패턴 매칭 (우선)
      Step 2: 파서별 TYPE_MAP 직접 매핑
      Step 3: plain_text 폴백
    """
    t = (text or "").strip()

    # Step 1: 텍스트 패턴 (파서 무관)
    if UNIT_RE.match(t):
        return "unit_label"
    if BRACKET_TITLE_RE.match(t):
        return "topic_title"

    # Step 2: 파서별 TYPE_MAP
    key = raw_label.lower().strip()
    mapped = type_map.get(key)
    if mapped:
        return mapped

    # Step 3: 폴백
    return "plain_text"


# ════════════════════════════════════════════════════════════════════
# 특허 핵심 1-2: 좌표 정규화
# ════════════════════════════════════════════════════════════════════
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def norm_bbox(bbox_xyxy, page_w: float, page_h: float) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    if page_w <= 0 or page_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        clamp01(x1 / page_w), clamp01(y1 / page_h),
        clamp01(x2 / page_w), clamp01(y2 / page_h),
    )


def poly_to_xyxy(poly: list) -> Tuple[float, float, float, float]:
    """8-value polygon [x1,y1,x2,y2,...,x4,y4] → (min_x, min_y, max_x, max_y)"""
    if not poly or len(poly) < 4:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(poly[i]) for i in range(0, len(poly), 2)]
    ys = [float(poly[i]) for i in range(1, len(poly), 2)]
    return (min(xs), min(ys), max(xs), max(ys))


# ════════════════════════════════════════════════════════════════════
# 어댑터 1: OmniDocBench GT JSON → List[CanonNode]
# ════════════════════════════════════════════════════════════════════
def omnidocbench_to_canon_nodes(
    page_entry: dict,
    image_name: str,
    text_model: SentenceTransformer,
) -> List[CanonNode]:
    """
    OmniDocBench 한 페이지 엔트리 → CanonNode 리스트

    page_entry 구조:
      {
        "layout_dets": [ { "category_type", "poly", "text", "order", ... } ],
        "page_info": { "height", "width", "image_path", ... }
      }
    """
    page_info = page_entry.get("page_info", {})
    pw = float(page_info.get("width", 1000))
    ph = float(page_info.get("height", 1000))
    page_key = image_name
    doc_key = hashlib.sha256(image_name.encode()).hexdigest()[:16]

    nodes = []
    layout_dets = page_entry.get("layout_dets", [])

    # ignore/mask 요소 필터링
    skip_types = {"abandon", "need_mask", "table_mask", "text_mask"}

    for det in layout_dets:
        cat = det.get("category_type", "unknown")
        if cat in skip_types:
            continue
        if det.get("ignore", False):
            continue

        poly = det.get("poly", [])
        bbox_abs = poly_to_xyxy(poly)
        bbox_n = norm_bbox(bbox_abs, pw, ph)

        text = det.get("text", "")
        order = det.get("order", -1)
        if order is None:
            order = -1

        canon_role = assign_canon_role(cat, text, OMNIDOCBENCH_TYPE_MAP)

        node_id = f"omni_{image_name}_{det.get('anno_id', uuid.uuid4().hex[:8])}"

        nodes.append(CanonNode(
            node_id=node_id,
            observed_type=cat,
            canon_role=canon_role,
            text=text or "",
            bbox=bbox_n,
            page=page_key,
            doc=doc_key,
            order=int(order),
            parser_score=1.0,  # GT
        ))

    # 텍스트 임베딩 일괄 계산
    texts = [n.text for n in nodes]
    if texts and text_model is not None:
        embs = text_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for i, n in enumerate(nodes):
            n.text_emb = embs[i].tolist()

    return nodes


# ════════════════════════════════════════════════════════════════════
# 어댑터 2: PaddleOCR-VL 1.5 raw JSON → List[CanonNode]
# ════════════════════════════════════════════════════════════════════
def _normalize_box_from_any(obj):
    """여러 형태의 bbox/poly를 [x1,y1,x2,y2]로 정규화"""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, (list, tuple)):
        if len(obj) == 4 and all(isinstance(v, (int, float)) for v in obj):
            return [float(v) for v in obj]
        if len(obj) >= 4 and isinstance(obj[0], (list, tuple)):
            pts = np.array(obj, dtype=np.float32)
            return [float(pts[:, 0].min()), float(pts[:, 1].min()),
                    float(pts[:, 0].max()), float(pts[:, 1].max())]
    return None


def _guess_label(elem: dict) -> str:
    label = (elem.get("label") or elem.get("type") or elem.get("category")
             or elem.get("cls") or elem.get("name") or elem.get("block_label")
             or elem.get("block_type"))
    if label is not None:
        return str(label).lower()
    content = elem.get("content") or elem.get("text") or elem.get("markdown") or ""
    if isinstance(content, str):
        low = content.lower()
        if "<table" in low or "</table>" in low:
            return "table"
        if "\\begin{" in low or "\\frac" in low:
            return "formula"
    return "unknown"


def _extract_layout_items(res_dict: dict) -> list:
    """PaddleOCR VL raw json에서 layout items 추출 (pp-vl.py의 extract_layout_items 재사용)"""
    items = []
    seen = set()

    def add(label, bbox, score, text):
        nb = _normalize_box_from_any(bbox)
        if nb is None:
            return
        key = (label, tuple(nb))
        if key not in seen:
            seen.add(key)
            items.append({"label": label, "bbox": nb, "score": score, "text": text})

    def parse_list(data_list):
        for elem in data_list:
            if not isinstance(elem, dict):
                continue
            label = _guess_label(elem)
            bbox = (elem.get("bbox") or elem.get("box") or elem.get("block_bbox")
                    or elem.get("poly") or elem.get("polygon") or elem.get("points"))
            score = elem.get("score")
            text = elem.get("text") or elem.get("content") or elem.get("markdown")
            add(label, bbox, score, text)

    for key in ["layout_det_res", "layout_result", "layout_results", "layout_res",
                "layout", "parsing_res_list", "parsing_result", "parsing_results",
                "overall_ocr_res", "doc_parsing_res"]:
        val = res_dict.get(key)
        if isinstance(val, list):
            parse_list(val)
        elif isinstance(val, dict):
            for subv in val.values():
                if isinstance(subv, list):
                    parse_list(subv)

    for k, v in res_dict.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            sample_keys = set(v[0].keys())
            if {"bbox", "block_bbox", "poly", "polygon"} & sample_keys:
                parse_list(v)

    return items


def paddleocr_vl_to_canon_nodes(
    raw_json: dict,
    image_name: str,
    image_w: int,
    image_h: int,
    text_model: SentenceTransformer,
) -> List[CanonNode]:
    """
    PaddleOCR-VL 1.5 raw JSON → CanonNode 리스트

    특허 파서 어댑터 계층:
      1. layout items 추출
      2. bbox 정규화 (pixel → 0~1)
      3. Canonical Role 매핑 (PADDLEOCR_VL_TYPE_MAP)
      4. Y→X 순으로 reading_order 자동 부여
      5. 텍스트 임베딩 생성
    """
    layout_items = _extract_layout_items(raw_json)

    page_key = image_name
    doc_key = hashlib.sha256(image_name.encode()).hexdigest()[:16]
    pw = float(image_w) if image_w > 0 else 1.0
    ph = float(image_h) if image_h > 0 else 1.0

    nodes = []
    for i, item in enumerate(layout_items):
        bbox_abs = item["bbox"]
        # bbox가 이미 0~1이면 그대로, 아니면 정규화
        if all(v <= 1.0 for v in bbox_abs):
            bbox_n = tuple(clamp01(v) for v in bbox_abs)
        else:
            bbox_n = norm_bbox(bbox_abs, pw, ph)

        raw_label = item["label"]
        text = item.get("text") or ""
        canon_role = assign_canon_role(raw_label, text, PADDLEOCR_VL_TYPE_MAP)

        score = float(item["score"]) if item.get("score") is not None else 0.7

        node_id = f"paddle_{image_name}_{i}"
        nodes.append(CanonNode(
            node_id=node_id,
            observed_type=raw_label,
            canon_role=canon_role,
            text=text,
            bbox=bbox_n,
            page=page_key,
            doc=doc_key,
            order=-1,  # 아래서 자동 부여
            parser_score=score,
        ))

    # reading_order 자동 부여 (Y→X 순)
    nodes.sort(key=lambda n: (n.bbox[1], n.bbox[0]))
    for i, n in enumerate(nodes):
        n.order = i + 1

    # 텍스트 임베딩 일괄 계산
    texts = [n.text for n in nodes]
    if texts and text_model is not None:
        embs = text_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for i, n in enumerate(nodes):
            n.text_emb = embs[i].tolist()

    return nodes


# ════════════════════════════════════════════════════════════════════
# 어댑터 3: Docling JSON → List[CanonNode]
# ════════════════════════════════════════════════════════════════════

DOCLING_TYPE_MAP = {
    "section_header": "section_header",
    "title":          "section_header",
    "heading":        "section_header",
    "paragraph":      "support_paragraph",
    "text":           "support_paragraph",
    "list_item":      "support_paragraph",
    "caption":        "support_paragraph",
    "footnote":       "support_paragraph",
    "page_footer":    "page_header",
    "page_header":    "page_header",
    "table":          "table",
    "picture":        "picture",
    "figure":         "picture",
    "chart":          "chart",
    "formula":        "support_paragraph",
    "code":           "support_paragraph",
    "reference":      "support_paragraph",
}


def _docling_bbox_to_xyxy(bbox: dict, page_h: float) -> Tuple[float, float, float, float]:
    """
    Docling bbox (BOTTOMLEFT origin) → (x1, y1, x2, y2) TOPLEFT origin
      BOTTOMLEFT: l, b = 좌하단 / r, t = 우상단
      TOPLEFT 변환: y_top = page_h - t_bottomleft
    """
    l = float(bbox.get("l", 0))
    b = float(bbox.get("b", 0))
    r = float(bbox.get("r", 0))
    t = float(bbox.get("t", 0))
    # BOTTOMLEFT → TOPLEFT
    y1 = page_h - t   # top in TOPLEFT = page_h - top_in_BOTTOMLEFT
    y2 = page_h - b
    return (l, y1, r, y2)


def docling_to_canon_nodes(
    doc_dict: dict,
    image_name: str,
    text_model,
) -> List[CanonNode]:
    """
    Docling export_to_dict() 결과 → CanonNode 리스트

    doc_dict 구조:
      {
        "pages": {"1": {"size": {"width": W, "height": H}}},
        "texts":    [{"label", "text", "prov": [{"bbox", "page_no"}]}],
        "tables":   [{"prov": [...], "data": {...}}],
        "pictures": [{"prov": [...]}],
      }

    좌표계: Docling은 BOTTOMLEFT origin → TOPLEFT로 변환 후 정규화
    reading_order: prov의 page_no → bbox y 순으로 자동 부여
    """
    page_key = image_name
    doc_key = hashlib.sha256(image_name.encode()).hexdigest()[:16]

    # 페이지별 크기
    pages_meta = doc_dict.get("pages", {})
    def _get_page_size(page_no: int):
        info = pages_meta.get(str(page_no)) or pages_meta.get(page_no, {})
        sz = info.get("size", {})
        return float(sz.get("width", 1000)), float(sz.get("height", 1000))

    nodes = []
    node_counter = 0

    def _add_node(label: str, text: str, prov_list: list):
        nonlocal node_counter
        if not prov_list:
            return
        prov = prov_list[0]
        page_no = prov.get("page_no", 1)
        pw, ph = _get_page_size(page_no)
        bbox_raw = prov.get("bbox", {})
        bbox_abs = _docling_bbox_to_xyxy(bbox_raw, ph)
        bbox_n = norm_bbox(bbox_abs, pw, ph)
        canon_role = assign_canon_role(label, text or "", DOCLING_TYPE_MAP)
        node_id = f"docling_{image_name}_{node_counter}"
        node_counter += 1
        nodes.append(CanonNode(
            node_id=node_id,
            observed_type=label,
            canon_role=canon_role,
            text=text or "",
            bbox=bbox_n,
            page=page_key,
            doc=doc_key,
            order=-1,   # 아래서 Y→X 순으로 자동 부여
            parser_score=0.9,
        ))

    # texts (section_header, paragraph, list_item, caption 등)
    for item in doc_dict.get("texts", []):
        _add_node(
            label=item.get("label", "text"),
            text=item.get("text", ""),
            prov_list=item.get("prov", []),
        )

    # tables (텍스트는 markdown으로 직렬화)
    for item in doc_dict.get("tables", []):
        # table 텍스트: grid cell 값 concat
        try:
            grid = item.get("data", {}).get("grid", [])
            cell_texts = [
                cell.get("text", "")
                for row in grid for cell in row
                if cell.get("text", "").strip()
            ]
            table_text = " | ".join(cell_texts)
        except Exception:
            table_text = ""
        _add_node(label="table", text=table_text, prov_list=item.get("prov", []))

    # pictures
    for item in doc_dict.get("pictures", []):
        caption_items = item.get("captions", [])
        caption_text = " ".join(c.get("text", "") for c in caption_items if isinstance(c, dict))
        _add_node(label="picture", text=caption_text, prov_list=item.get("prov", []))

    # reading_order 자동 부여 (Y→X 순)
    nodes.sort(key=lambda n: (n.bbox[1], n.bbox[0]))
    for i, n in enumerate(nodes):
        n.order = i + 1

    # 텍스트 임베딩 일괄 계산
    texts = [n.text for n in nodes]
    if texts and text_model is not None:
        embs = text_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for i, n in enumerate(nodes):
            n.text_emb = embs[i].tolist()

    return nodes


# ════════════════════════════════════════════════════════════════════
# 어댑터 4: MinerU (magic-pdf) middle.json → List[CanonNode]
# ════════════════════════════════════════════════════════════════════

MINERU_TYPE_MAP = {
    "text":              "support_paragraph",
    "title":             "section_header",
    "interline_equation":"support_paragraph",
    "image":             "picture",
    "image_body":        "picture",
    "image_caption":     "support_paragraph",
    "image_footnote":    "support_paragraph",
    "table":             "table",
    "table_body":        "table",
    "table_caption":     "support_paragraph",
    "table_footnote":    "support_paragraph",
    "footnote":          "support_paragraph",
    "list":              "support_paragraph",
    "index":             "support_paragraph",
    "discarded":         "plain_text",
}


def _mineru_extract_text(para_block: dict) -> str:
    """MinerU para_block에서 텍스트 추출 (lines → spans → content)."""
    text_parts = []
    for line in para_block.get("lines", []):
        for span in line.get("spans", []):
            content = span.get("content", "") or span.get("text", "")
            if content and span.get("type") in ("text", "inline_equation"):
                text_parts.append(content.strip())
    return " ".join(text_parts)


def mineru_to_canon_nodes(
    middle_json: dict,
    image_name: str,
    text_model,
    page_idx: int = 0,
) -> List[CanonNode]:
    """
    MinerU *_middle.json → CanonNode 리스트

    middle_json 구조:
      {
        "pdf_info": [
          {
            "page_size": [width, height],
            "para_blocks": [
              {"type": "text"|"title"|"image"|"table"|...,
               "bbox": [x0, y0, x1, y1],
               "lines": [{"spans": [{"type": "text", "content": "..."}]}],
               "blocks": [...]   # image/table compound blocks
              }
            ]
          }
        ]
      }

    좌표계: MinerU는 TOPLEFT origin, pixel 단위
    """
    page_key = image_name
    doc_key = hashlib.sha256(image_name.encode()).hexdigest()[:16]

    pdf_info = middle_json.get("pdf_info", [])
    if not pdf_info or page_idx >= len(pdf_info):
        return []

    page_data = pdf_info[page_idx]
    page_size = page_data.get("page_size", [1000, 1000])
    pw = float(page_size[0]) if page_size[0] > 0 else 1000.0
    ph = float(page_size[1]) if page_size[1] > 0 else 1000.0

    nodes = []
    node_counter = 0

    def _add_block(block_type: str, bbox: list, text: str):
        nonlocal node_counter
        if not bbox or len(bbox) < 4:
            return
        x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        bbox_n = norm_bbox((x0, y0, x1, y1), pw, ph)
        canon_role = assign_canon_role(block_type, text, MINERU_TYPE_MAP)
        node_id = f"mineru_{image_name}_{node_counter}"
        node_counter += 1
        nodes.append(CanonNode(
            node_id=node_id,
            observed_type=block_type,
            canon_role=canon_role,
            text=text or "",
            bbox=bbox_n,
            page=page_key,
            doc=doc_key,
            order=-1,
            parser_score=0.85,
        ))

    for para_block in page_data.get("para_blocks", []):
        ptype = para_block.get("type", "text")
        pbbox = para_block.get("bbox", [])

        if ptype in ("image", "table"):
            # compound block: iterate sub-blocks
            for sub in para_block.get("blocks", []):
                stype = sub.get("type", ptype)
                sbbox = sub.get("bbox", pbbox)
                if stype in ("image_body", "table_body"):
                    _add_block(stype, sbbox, "")
                else:
                    stext = _mineru_extract_text(sub)
                    _add_block(stype, sbbox, stext)
        else:
            text = _mineru_extract_text(para_block)
            _add_block(ptype, pbbox, text)

    # reading_order 자동 부여 (Y→X 순)
    nodes.sort(key=lambda n: (n.bbox[1], n.bbox[0]))
    for i, n in enumerate(nodes):
        n.order = i + 1

    # 텍스트 임베딩 일괄 계산
    texts = [n.text for n in nodes]
    if texts and text_model is not None:
        embs = text_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for i, n in enumerate(nodes):
            n.text_emb = embs[i].tolist()

    return nodes


# ════════════════════════════════════════════════════════════════════
# main.py EU 알고리즘 (Neo4j 없이 독립 실행 버전)
# ════════════════════════════════════════════════════════════════════

# ── helpers (main.py 동일) ──
def vertical_gap(a: CanonNode, b: CanonNode) -> float:
    upper, lower = (a, b) if a.bbox[1] <= b.bbox[1] else (b, a)
    return max(0.0, lower.bbox[1] - upper.bbox[3])


def left_x_diff(a: CanonNode, b: CanonNode) -> float:
    return abs(a.bbox[0] - b.bbox[0])


def is_visual(n: CanonNode) -> bool:
    return n.canon_role in {"table", "chart", "picture"}


def is_header_anchor(n: CanonNode) -> bool:
    return n.canon_role in {"section_header", "topic_title"}


def sorted_page_nodes(canon_nodes: List[CanonNode]) -> Dict[str, List[CanonNode]]:
    by_page = defaultdict(list)
    for n in canon_nodes:
        by_page[n.page].append(n)
    for page in by_page:
        by_page[page].sort(key=lambda x: (x.order if x.order >= 0 else 10**9, x.bbox[1], x.bbox[0]))
    return by_page


def eu_kind_from_roles(roles: List[str], has_header_anchor: bool = False) -> str:
    role_set = set(roles)
    if role_set == {"plain_text"} or role_set <= {"plain_text", "support_paragraph"}:
        if "plain_text" in role_set and not (role_set & {"table", "chart", "picture"}):
            return "text_cluster_panel"
    if "table" in role_set and "chart" in role_set:
        return "stat_panel"
    if "table" in role_set:
        return "table_panel"
    if "chart" in role_set:
        return "chart_panel"
    if "picture" in role_set:
        return "visual_panel"
    if has_header_anchor:
        return "section_text_panel"
    return "text_cluster_panel"


def maybe_add_text_neighbor(anchor: CanonNode, cand: CanonNode, member_ids: Set[str]) -> bool:
    if cand.node_id in member_ids:
        return False
    if cand.page != anchor.page:
        return False
    if cand.order >= 0 and anchor.order >= 0:
        if abs(cand.order - anchor.order) > Config.HEADER_ATTACH_MAX_ORDER_GAP:
            return False
    if vertical_gap(anchor, cand) > Config.TEXT_CLUSTER_GAP_Y * 1.5:
        return False
    if left_x_diff(anchor, cand) > Config.TEXT_CLUSTER_X_ALIGN_TAU * 1.5:
        return False
    return True


# ── Phase A + Phase B (main.py materialize_visual_page_eus 동일) ──
def materialize_visual_eus(
    canon_nodes: List[CanonNode],
    start_idx: int = 1,
    paragraph_semantic_tau: float = 0.40,
) -> Tuple[List[MaterializedEU], Set[str]]:
    nodes_by_id = {n.node_id: n for n in canon_nodes}
    by_page = sorted_page_nodes(canon_nodes)
    STRUCTURAL_ROLES = {"section_header", "topic_title", "unit_label"}

    eus: List[MaterializedEU] = []
    eu_idx = start_idx
    covered: Set[str] = set()
    visited_visual: Set[str] = set()

    for page, page_nodes in by_page.items():
        visuals = [n for n in page_nodes if is_visual(n)]

        # nearest_visual_map
        nearest_visual_map: Dict[str, str] = {}
        for n in page_nodes:
            if n.canon_role not in STRUCTURAL_ROLES:
                continue
            best_vid, best_dist = None, float("inf")
            for v in visuals:
                dist = vertical_gap(v, n) + left_x_diff(v, n) * 0.3
                if dist < best_dist:
                    best_dist = dist
                    best_vid = v.node_id
            if best_vid is not None and best_dist < 0.30:
                nearest_visual_map[n.node_id] = best_vid

        # Phase A
        page_eus: List[MaterializedEU] = []
        for seed in visuals:
            if seed.node_id in visited_visual:
                continue
            member_ids: Set[str] = {seed.node_id}
            eu_visual_ids = {seed.node_id}
            seed_order = seed.order if seed.order >= 0 else 10**9

            for cand in page_nodes:
                if cand.node_id in member_ids or cand.canon_role not in STRUCTURAL_ROLES:
                    continue
                if cand.order >= 0 and cand.order < seed_order - Config.HEADER_ATTACH_MAX_ORDER_GAP:
                    continue
                nearest_v = nearest_visual_map.get(cand.node_id)
                if nearest_v is not None and nearest_v not in eu_visual_ids:
                    continue
                anchor = nodes_by_id.get(nearest_v, seed) if nearest_v else seed
                if maybe_add_text_neighbor(anchor, cand, member_ids):
                    member_ids.add(cand.node_id)

            for mid in list(member_ids):
                if is_visual(nodes_by_id[mid]):
                    visited_visual.add(mid)
            covered.update(member_ids)
            roles = [nodes_by_id[mid].canon_role for mid in member_ids]

            eu = MaterializedEU(
                eu_id=f"EU-{eu_idx:04d}", kind=eu_kind_from_roles(roles),
                page=page, primary_id=seed.node_id,
                member_node_ids=sorted(member_ids), edge_ids=[],
                meta={"primary_role": seed.canon_role, "member_roles": roles, "source": "visual_seed"},
            )
            page_eus.append(eu)
            eus.append(eu)
            eu_idx += 1

        # Phase B: Semantic Paragraph Attachment
        if page_eus:
            unattached = [
                n for n in page_nodes
                if n.canon_role == "support_paragraph" and n.node_id not in covered
            ]
            for para in unattached:
                best_eu_idx, best_sim = -1, 0.0
                for ei, eu in enumerate(page_eus):
                    for mid in eu.member_node_ids:
                        m = nodes_by_id.get(mid)
                        if m is None:
                            continue
                        sim = cosine_sim(para.text_emb, m.text_emb)
                        if sim is not None and sim > best_sim:
                            best_sim = sim
                            best_eu_idx = ei
                if best_sim >= paragraph_semantic_tau and best_eu_idx >= 0:
                    best_eu = page_eus[best_eu_idx]
                    best_eu.member_node_ids.append(para.node_id)
                    best_eu.meta["member_roles"].append("support_paragraph")
                    covered.add(para.node_id)

    return eus, covered


# ── Phase C-1: header text zones (main.py 동일) ──
def materialize_header_text_eus(
    canon_nodes: List[CanonNode],
    already_covered: Set[str],
    start_idx: int,
) -> Tuple[List[MaterializedEU], Set[str]]:
    by_page = sorted_page_nodes(canon_nodes)
    nodes_by_id = {n.node_id: n for n in canon_nodes}
    eus, covered_now = [], set()
    eu_idx = start_idx

    for page, page_nodes in by_page.items():
        eligible = [
            n for n in page_nodes
            if n.node_id not in already_covered
            and n.canon_role in {"section_header", "topic_title", "support_paragraph", "unit_label", "plain_text"}
        ]
        if not eligible:
            continue

        i = 0
        while i < len(eligible):
            anchor = eligible[i]
            if not is_header_anchor(anchor):
                i += 1
                continue
            member_ids: Set[str] = {anchor.node_id}
            last_added = anchor
            j = i + 1
            while j < len(eligible):
                cand = eligible[j]
                if cand.node_id in already_covered:
                    j += 1
                    continue
                if is_visual(cand) or is_header_anchor(cand):
                    break
                if cand.canon_role not in {"support_paragraph", "unit_label"}:
                    break
                if cand.canon_role == "unit_label":
                    if left_x_diff(anchor, cand) > Config.TEXT_CLUSTER_X_ALIGN_TAU * 3:
                        j += 1
                        continue
                if anchor.order >= 0 and cand.order >= 0:
                    if cand.order - anchor.order > Config.HEADER_ATTACH_MAX_ORDER_GAP:
                        break
                if vertical_gap(last_added, cand) > Config.TEXT_CLUSTER_GAP_Y * 1.7:
                    break
                member_ids.add(cand.node_id)
                last_added = cand
                j += 1

            roles = [nodes_by_id[mid].canon_role for mid in member_ids]
            eus.append(MaterializedEU(
                eu_id=f"EU-{eu_idx:04d}", kind=eu_kind_from_roles(roles, has_header_anchor=True),
                page=page, primary_id=anchor.node_id,
                member_node_ids=sorted(member_ids), edge_ids=[],
                meta={"primary_role": anchor.canon_role, "member_roles": roles, "source": "header_anchor"},
            ))
            covered_now.update(member_ids)
            eu_idx += 1
            i = j

    return eus, covered_now


# ── Phase C-3: text cluster fallback (main.py 동일) ──
def can_join_text_cluster(prev: CanonNode, cur: CanonNode) -> bool:
    if prev.page != cur.page:
        return False
    if is_header_anchor(cur) or is_header_anchor(prev):
        return False
    if cur.canon_role not in {"support_paragraph", "unit_label", "plain_text"}:
        return False
    if prev.canon_role not in {"support_paragraph", "unit_label", "plain_text"}:
        return False
    if prev.order >= 0 and cur.order >= 0:
        if abs(cur.order - prev.order) > Config.TEXT_CLUSTER_ORDER_GAP:
            return False
    if vertical_gap(prev, cur) > Config.TEXT_CLUSTER_GAP_Y:
        return False
    if left_x_diff(prev, cur) > Config.TEXT_CLUSTER_X_ALIGN_TAU:
        return False
    return True


def materialize_text_cluster_eus(
    canon_nodes: List[CanonNode],
    already_covered: Set[str],
    start_idx: int,
) -> Tuple[List[MaterializedEU], Set[str]]:
    by_page = sorted_page_nodes(canon_nodes)
    nodes_by_id = {n.node_id: n for n in canon_nodes}
    eus, covered_now = [], set()
    eu_idx = start_idx

    for page, page_nodes in by_page.items():
        leftovers = [
            n for n in page_nodes
            if n.node_id not in already_covered
            and n.canon_role in {"support_paragraph", "unit_label", "plain_text"}
        ]
        if not leftovers:
            continue

        cluster: List[CanonNode] = []
        for n in leftovers:
            if not cluster:
                cluster = [n]
                continue
            if can_join_text_cluster(cluster[-1], n):
                cluster.append(n)
            else:
                member_ids = {x.node_id for x in cluster}
                roles = [nodes_by_id[mid].canon_role for mid in member_ids]
                eus.append(MaterializedEU(
                    eu_id=f"EU-{eu_idx:04d}", kind=eu_kind_from_roles(roles),
                    page=page, primary_id=cluster[0].node_id,
                    member_node_ids=sorted(member_ids), edge_ids=[],
                    meta={"primary_role": cluster[0].canon_role, "member_roles": roles, "source": "text_cluster"},
                ))
                covered_now.update(member_ids)
                eu_idx += 1
                cluster = [n]

        if cluster:
            member_ids = {x.node_id for x in cluster}
            roles = [nodes_by_id[mid].canon_role for mid in member_ids]
            eus.append(MaterializedEU(
                eu_id=f"EU-{eu_idx:04d}", kind=eu_kind_from_roles(roles),
                page=page, primary_id=cluster[0].node_id,
                member_node_ids=sorted(member_ids), edge_ids=[],
                meta={"primary_role": cluster[0].canon_role, "member_roles": roles, "source": "text_cluster"},
            ))
            covered_now.update(member_ids)
            eu_idx += 1

    return eus, covered_now


# ════════════════════════════════════════════════════════════════════
# 전체 EU 파이프라인
# ════════════════════════════════════════════════════════════════════
def run_eu_pipeline(canon_nodes: List[CanonNode]) -> List[MaterializedEU]:
    """
    main.py와 동일한 EU 파이프라인:
      Phase A: Visual-Core EU (구조적 요소 proximity 부착)
      Phase B: Semantic Paragraph Attachment (유사도 행렬 전역 배정)
      Phase C-1: Header-anchored text zones
      Phase C-3: Text cluster fallback
    """
    # Phase A + B
    visual_eus, covered = materialize_visual_eus(canon_nodes)
    logger.info(f"  Phase A+B: {len(visual_eus)} visual EUs, {len(covered)} covered nodes")

    next_idx = len(visual_eus) + 1

    # Phase C-1: header text zones
    header_eus, header_covered = materialize_header_text_eus(canon_nodes, covered, next_idx)
    covered.update(header_covered)
    next_idx += len(header_eus)
    logger.info(f"  Phase C-1: {len(header_eus)} header EUs, {len(header_covered)} covered nodes")

    # Phase C-3: text cluster fallback
    cluster_eus, cluster_covered = materialize_text_cluster_eus(canon_nodes, covered, next_idx)
    covered.update(cluster_covered)
    logger.info(f"  Phase C-3: {len(cluster_eus)} cluster EUs, {len(cluster_covered)} covered nodes")

    all_eus = visual_eus + header_eus + cluster_eus

    # 최종 정렬 + EU ID 재부여
    nodes_by_id = {n.node_id: n for n in canon_nodes}

    def _sort_key(eu):
        orders = [nodes_by_id[m].order for m in eu.member_node_ids if m in nodes_by_id and nodes_by_id[m].order >= 0]
        min_order = min(orders) if orders else 10**9
        ys = [nodes_by_id[m].bbox[1] for m in eu.member_node_ids if m in nodes_by_id]
        avg_y = sum(ys) / len(ys) if ys else 0.0
        return (eu.page, min_order, avg_y)

    all_eus.sort(key=_sort_key)
    for i, eu in enumerate(all_eus, 1):
        eu.eu_id = f"EU-{i:04d}"

    uncovered = [n for n in canon_nodes if n.node_id not in covered]
    logger.info(f"  Total: {len(all_eus)} EUs, {len(covered)}/{len(canon_nodes)} nodes covered, "
                f"{len(uncovered)} uncovered")

    return all_eus


# ════════════════════════════════════════════════════════════════════
# EU → JSON 직렬화
# ════════════════════════════════════════════════════════════════════
def eu_to_dict(eu: MaterializedEU, nodes_by_id: Dict[str, CanonNode]) -> dict:
    elements = []
    for mid in eu.member_node_ids:
        n = nodes_by_id.get(mid)
        if n is None:
            continue
        elements.append({
            "id": n.node_id,
            "observed_type": n.observed_type,
            "canon_role": n.canon_role,
            "text": n.text,
            "bbox": list(n.bbox),
            "order": n.order,
        })
    elements.sort(key=lambda e: e["order"] if e["order"] >= 0 else 10**9)

    return {
        "eu_id": eu.eu_id,
        "kind": eu.kind,
        "page": eu.page,
        "primary_id": eu.primary_id,
        "source": eu.meta.get("source", ""),
        "member_count": len(eu.member_node_ids),
        "elements": elements,
    }


# ════════════════════════════════════════════════════════════════════
# w/o EU 출력 (비교용: 정규화만 하고 EU 없이 개별 요소 그대로)
# ════════════════════════════════════════════════════════════════════
def nodes_to_chunks_without_eu(canon_nodes: List[CanonNode]) -> List[dict]:
    """EU 없이 개별 요소를 그대로 chunk로 출력 (w/o EU 베이스라인)"""
    chunks = []
    for n in canon_nodes:
        if n.canon_role == "page_header":
            continue
        chunks.append({
            "chunk_id": n.node_id,
            "text": n.text,
            "canon_role": n.canon_role,
            "bbox": list(n.bbox),
            "page": n.page,
            "order": n.order,
        })
    chunks.sort(key=lambda c: (c["page"], c["order"] if c["order"] >= 0 else 10**9))
    return chunks


# ════════════════════════════════════════════════════════════════════
# Main CLI
# ════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Parser → Canonical Role → EU Pipeline")
    parser.add_argument("--source", choices=["omnidocbench", "paddleocr_vl", "docling", "mineru"], required=True)
    parser.add_argument("--gt", type=str, help="OmniDocBench GT JSON path")
    parser.add_argument("--input-dir", type=str, help="PaddleOCR VL raw JSON directory")
    parser.add_argument("--images", type=str, help="Image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Max pages to process (0=all)")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding computation")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "with_eu").mkdir(exist_ok=True)
    (output_dir / "without_eu").mkdir(exist_ok=True)

    # 모델 로드
    text_model = None
    if not args.no_embed:
        logger.info("Loading text embedding model (ko-sbert)...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        text_model = SentenceTransformer(Config.ST_STRUCTURE_PATH)
        text_model.load_state_dict(torch.load(Config.TEXT_EMBED_PATH, map_location="cpu"))
        text_model.eval()
        logger.info("Model loaded.")

    if args.source == "omnidocbench":
        _run_omnidocbench(args, text_model, output_dir)
    elif args.source == "paddleocr_vl":
        _run_paddleocr_vl(args, text_model, output_dir)
    elif args.source == "docling":
        _run_docling(args, text_model, output_dir)
    elif args.source == "mineru":
        _run_mineru(args, text_model, output_dir)


def _run_omnidocbench(args, text_model, output_dir: Path):
    gt_path = Path(args.gt)
    logger.info(f"Loading OmniDocBench GT: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    logger.info(f"Total pages: {len(pages)}")

    if args.limit > 0:
        pages = pages[:args.limit]

    stats = {"total_pages": 0, "total_nodes": 0, "total_eus": 0, "eu_kinds": defaultdict(int)}

    for idx, page_entry in enumerate(pages):
        page_info = page_entry.get("page_info", {})
        image_path = page_info.get("image_path", f"page_{idx}")
        image_name = Path(image_path).stem

        logger.info(f"[{idx+1}/{len(pages)}] {image_name}")

        # 정규화: OmniDocBench GT → CanonNode
        canon_nodes = omnidocbench_to_canon_nodes(page_entry, image_name, text_model)
        if not canon_nodes:
            logger.warning(f"  No nodes for {image_name}")
            continue

        # w/o EU
        chunks = nodes_to_chunks_without_eu(canon_nodes)
        wo_path = output_dir / "without_eu" / f"{image_name}.json"
        with open(wo_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # w/ EU
        eus = run_eu_pipeline(canon_nodes)
        nodes_by_id = {n.node_id: n for n in canon_nodes}
        eu_dicts = [eu_to_dict(eu, nodes_by_id) for eu in eus]
        w_path = output_dir / "with_eu" / f"{image_name}.json"
        with open(w_path, "w", encoding="utf-8") as f:
            json.dump(eu_dicts, f, ensure_ascii=False, indent=2)

        stats["total_pages"] += 1
        stats["total_nodes"] += len(canon_nodes)
        stats["total_eus"] += len(eus)
        for eu in eus:
            stats["eu_kinds"][eu.kind] += 1

    # 통계 저장
    stats["eu_kinds"] = dict(stats["eu_kinds"])
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Stats: {json.dumps(stats, ensure_ascii=False)}")


def _run_paddleocr_vl(args, text_model, output_dir: Path):
    input_dir = Path(args.input_dir)
    image_dir = Path(args.images) if args.images else None

    json_files = sorted(input_dir.glob("*.json"))
    if args.limit > 0:
        json_files = json_files[:args.limit]
    logger.info(f"PaddleOCR VL JSON files: {len(json_files)}")

    stats = {"total_pages": 0, "total_nodes": 0, "total_eus": 0, "eu_kinds": defaultdict(int)}

    for idx, jf in enumerate(json_files):
        image_name = jf.stem
        logger.info(f"[{idx+1}/{len(json_files)}] {image_name}")

        with open(jf, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        # 이미지 크기 추론
        img_w, img_h = 1000, 1000
        if image_dir:
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                img_path = image_dir / f"{image_name}{ext}"
                if img_path.exists():
                    try:
                        import cv2
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_h, img_w = img.shape[:2]
                    except ImportError:
                        from PIL import Image
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size
                    break

        # 정규화: PaddleOCR VL → CanonNode
        canon_nodes = paddleocr_vl_to_canon_nodes(raw_json, image_name, img_w, img_h, text_model)
        if not canon_nodes:
            logger.warning(f"  No nodes for {image_name}")
            continue

        # w/o EU
        chunks = nodes_to_chunks_without_eu(canon_nodes)
        wo_path = output_dir / "without_eu" / f"{image_name}.json"
        with open(wo_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # w/ EU
        eus = run_eu_pipeline(canon_nodes)
        nodes_by_id = {n.node_id: n for n in canon_nodes}
        eu_dicts = [eu_to_dict(eu, nodes_by_id) for eu in eus]
        w_path = output_dir / "with_eu" / f"{image_name}.json"
        with open(w_path, "w", encoding="utf-8") as f:
            json.dump(eu_dicts, f, ensure_ascii=False, indent=2)

        stats["total_pages"] += 1
        stats["total_nodes"] += len(canon_nodes)
        stats["total_eus"] += len(eus)
        for eu in eus:
            stats["eu_kinds"][eu.kind] += 1

    stats["eu_kinds"] = dict(stats["eu_kinds"])
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Stats: {json.dumps(stats, ensure_ascii=False)}")


def _run_docling(args, text_model, output_dir: Path):
    """
    Docling으로 OmniDocBench 이미지를 파싱 → EU 생성.

    --images: OmniDocBench 이미지 디렉토리 (PNG/JPG)
    --english-only 여부는 호출 전 images 목록으로 제어

    사용법:
      python eu_from_parser.py --source docling \
          --images datasets/omnidocbench/source_hf/images \
          --output output/eu_docling \
          [--limit 10]
    """
    from docling.document_converter import DocumentConverter

    image_dir = Path(args.images) if args.images else None
    if image_dir is None or not image_dir.exists():
        logger.error("--images 디렉토리가 필요합니다.")
        return

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in exts)
    if args.limit > 0:
        image_files = image_files[:args.limit]
    logger.info(f"Docling: {len(image_files)} images to process")

    from PIL import Image as PILImage
    # Disable PIL decompression bomb limit for this process.
    # ocrmac (used inside Docling) also calls PIL.Image.open(), so the limit
    # must be lifted globally — not just in our probe code.
    PILImage.MAX_IMAGE_PIXELS = None

    conv = DocumentConverter()
    stats = {"total_pages": 0, "total_nodes": 0, "total_eus": 0, "eu_kinds": defaultdict(int)}

    for idx, img_path in enumerate(image_files):
        image_name = img_path.stem
        logger.info(f"[{idx+1}/{len(image_files)}] {image_name}")

        # Resume: skip already-completed images
        w_path = output_dir / "with_eu" / f"{image_name}.json"
        if w_path.exists():
            logger.info(f"  Already done, skipping.")
            continue

        try:
            result = conv.convert(str(img_path))
            doc_dict = result.document.export_to_dict()
        except Exception as e:
            logger.warning(f"  Docling 실패: {e}")
            continue

        canon_nodes = docling_to_canon_nodes(doc_dict, image_name, text_model)
        if not canon_nodes:
            logger.warning(f"  No nodes for {image_name}")
            continue

        # w/o EU
        chunks = nodes_to_chunks_without_eu(canon_nodes)
        wo_path = output_dir / "without_eu" / f"{image_name}.json"
        with open(wo_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # w/ EU
        eus = run_eu_pipeline(canon_nodes)
        nodes_by_id = {n.node_id: n for n in canon_nodes}
        eu_dicts = [eu_to_dict(eu, nodes_by_id) for eu in eus]
        w_path = output_dir / "with_eu" / f"{image_name}.json"
        with open(w_path, "w", encoding="utf-8") as f:
            json.dump(eu_dicts, f, ensure_ascii=False, indent=2)

        stats["total_pages"] += 1
        stats["total_nodes"] += len(canon_nodes)
        stats["total_eus"] += len(eus)
        for eu in eus:
            stats["eu_kinds"][eu.kind] += 1

    stats["eu_kinds"] = dict(stats["eu_kinds"])
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Stats: {json.dumps(stats, ensure_ascii=False)}")


def _run_mineru(args, text_model, output_dir: Path):
    """
    MinerU(magic-pdf CLI)로 OmniDocBench 이미지를 파싱 → EU 생성.

    각 이미지에 대해 `magic-pdf -p <img> -o <tmp> -m ocr` 를 subprocess로 호출하고,
    출력된 *_middle.json을 읽어 mineru_to_canon_nodes()로 변환한다.

    사전 조건: magic-pdf 모델 다운로드 완료
      conda run -n seu-eval mineru-models-download
      (또는 https://github.com/opendatalab/MinerU 참고)

    --images: OmniDocBench 이미지 디렉토리 (PNG/JPG)
    """
    import subprocess
    import tempfile

    image_dir = Path(args.images) if args.images else None
    if image_dir is None or not image_dir.exists():
        logger.error("--images 디렉토리가 필요합니다.")
        return

    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    image_files = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in exts)
    if args.limit > 0:
        image_files = image_files[:args.limit]
    logger.info(f"MinerU: {len(image_files)} images to process")

    stats = {"total_pages": 0, "total_nodes": 0, "total_eus": 0, "eu_kinds": defaultdict(int)}

    for idx, img_path in enumerate(image_files):
        image_name = img_path.stem
        logger.info(f"[{idx+1}/{len(image_files)}] {image_name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # magic-pdf CLI 호출
            cmd = [
                "magic-pdf",
                "-p", str(img_path),
                "-o", tmp_dir,
                "-m", "ocr",
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                if proc.returncode != 0:
                    logger.warning(f"  magic-pdf 실패 (rc={proc.returncode}): {proc.stderr[-500:]}")
                    continue
            except FileNotFoundError:
                logger.error("magic-pdf CLI를 찾을 수 없습니다. conda 환경을 확인하세요.")
                return
            except subprocess.TimeoutExpired:
                logger.warning(f"  magic-pdf 타임아웃: {image_name}")
                continue

            # 출력 middle.json 탐색
            # magic-pdf 출력 구조: {tmp_dir}/{image_name}/auto/{image_name}_middle.json
            middle_json_path = None
            for candidate in Path(tmp_dir).rglob("*_middle.json"):
                middle_json_path = candidate
                break

            if middle_json_path is None:
                logger.warning(f"  middle.json을 찾지 못했습니다: {image_name}")
                continue

            with open(middle_json_path, "r", encoding="utf-8") as f:
                middle_json = json.load(f)

        canon_nodes = mineru_to_canon_nodes(middle_json, image_name, text_model)
        if not canon_nodes:
            logger.warning(f"  No nodes for {image_name}")
            continue

        # w/o EU
        chunks = nodes_to_chunks_without_eu(canon_nodes)
        wo_path = output_dir / "without_eu" / f"{image_name}.json"
        with open(wo_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # w/ EU
        eus = run_eu_pipeline(canon_nodes)
        nodes_by_id = {n.node_id: n for n in canon_nodes}
        eu_dicts = [eu_to_dict(eu, nodes_by_id) for eu in eus]
        w_path = output_dir / "with_eu" / f"{image_name}.json"
        with open(w_path, "w", encoding="utf-8") as f:
            json.dump(eu_dicts, f, ensure_ascii=False, indent=2)

        stats["total_pages"] += 1
        stats["total_nodes"] += len(canon_nodes)
        stats["total_eus"] += len(eus)
        for eu in eus:
            stats["eu_kinds"][eu.kind] += 1

    stats["eu_kinds"] = dict(stats["eu_kinds"])
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Done. Stats: {json.dumps(stats, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
