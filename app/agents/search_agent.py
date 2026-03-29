"""의료 문서 하이브리드 검색 서브 에이전트.

edu-collection(BM25) + edu-medicine-info(Vector kNN) 병렬 검색 → 병합 → ReRank
파이프라인을 LangGraph StateGraph로 구성하고, @tool 래핑하여 medical_agent에 통합.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from elasticsearch import Elasticsearch
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── 검색 설정 ──────────────────────────────────────────────
_TOP_K = 5
_RERANK_MODEL = "rerank-v3.5"
_BM25_INDEX = settings.ES_INDEX       # edu-collection
_VECTOR_INDEX = "edu-medicine-info"
_CONTENT_FIELD = "content"
_VECTOR_FIELD = "content_vector"

# ── 싱글턴 클라이언트 ─────────────────────────────────────
_es: Elasticsearch | None = None
_emb: OpenAIEmbeddings | None = None
_co = None


def _es_client() -> Elasticsearch:
    global _es
    if _es is None:
        _es = Elasticsearch(
            settings.ES_URL,
            basic_auth=(settings.ES_USERNAME, settings.ES_PASSWORD),
            verify_certs=True,
        )
    return _es


def _embeddings() -> OpenAIEmbeddings:
    global _emb
    if _emb is None:
        _emb = OpenAIEmbeddings(model="text-embedding-3-small")
    return _emb


def _cohere():
    global _co
    if _co is None:
        key = getattr(settings, "COHERE_API_KEY", None)
        if not key:
            return None
        import cohere

        _co = cohere.Client(api_key=key)
    return _co


# ── State ──────────────────────────────────────────────────


class SearchState(TypedDict):
    query: str
    bm25_hits: list[dict]
    vector_hits: list[dict]
    merged: list[dict]
    result: str


# ── 노드 함수 ─────────────────────────────────────────────


def bm25_search(state: SearchState) -> dict:
    """BM25 키워드 검색 (edu-collection)"""
    try:
        resp = _es_client().search(
            index=_BM25_INDEX,
            body={
                "query": {"match": {_CONTENT_FIELD: {"query": state["query"], "operator": "or"}}},
                "size": _TOP_K,
            },
        )
        return {"bm25_hits": resp["hits"]["hits"]}
    except Exception as e:
        logger.warning("BM25 검색 실패: %s", e)
        return {"bm25_hits": []}


def vector_search(state: SearchState) -> dict:
    """Vector kNN 의미 검색 (edu-medicine-info)"""
    try:
        vec = _embeddings().embed_query(state["query"])
        resp = _es_client().search(
            index=_VECTOR_INDEX,
            body={
                "knn": {
                    "field": _VECTOR_FIELD,
                    "query_vector": vec,
                    "k": _TOP_K,
                    "num_candidates": _TOP_K * 10,
                },
                "size": _TOP_K,
            },
        )
        return {"vector_hits": resp["hits"]["hits"]}
    except Exception as e:
        logger.warning("Vector 검색 실패: %s", e)
        return {"vector_hits": []}


def merge_results(state: SearchState) -> dict:
    """BM25 + Vector 결과 병합, content 앞 200자 기준 중복 제거"""
    seen: set[str] = set()
    merged: list[dict] = []
    for hit in state["bm25_hits"] + state["vector_hits"]:
        key = hit.get("_source", {}).get(_CONTENT_FIELD, "")[:200]
        if key and key not in seen:
            seen.add(key)
            merged.append(hit)
    return {"merged": merged}


def _format_hit(rank: int, hit: dict, score_label: str) -> str:
    """검색 결과 한 건을 포맷팅"""
    src = hit["_source"]
    meta = src.get("metadata", {})
    content = src.get(_CONTENT_FIELD, "")[:500].replace("\n", " ")
    origin = meta.get("source", meta.get("source_spec", "unknown"))
    page = meta.get("page", "")
    header = f"[{rank}] {score_label} | 출처: {origin}"
    if page:
        header += f" (p{page})"
    return f"{header}\n{content}"


def rerank(state: SearchState) -> dict:
    """Cohere ReRank 적용, 미설정 시 score 정렬 fallback"""
    hits = state["merged"]
    query = state["query"]

    if not hits:
        return {"result": f"'{query}'에 대한 검색 결과를 찾을 수 없습니다."}

    # Cohere ReRank 시도
    co = _cohere()
    if co is not None:
        try:
            rr = co.rerank(
                query=query,
                documents=[h["_source"][_CONTENT_FIELD] for h in hits],
                model=_RERANK_MODEL,
                top_n=_TOP_K,
            )
            lines = [
                _format_hit(i, hits[r.index], f"relevance={r.relevance_score:.4f}")
                for i, r in enumerate(rr.results, 1)
            ]
            return {"result": f"■ 검색 결과 (BM25+Vector+ReRank, {len(lines)}건)\n\n" + "\n\n".join(lines)}
        except Exception as e:
            logger.warning("Cohere ReRank 실패, score 정렬로 fallback: %s", e)

    # Fallback: ES score 정렬
    ranked = sorted(hits, key=lambda h: h.get("_score", 0), reverse=True)[:_TOP_K]
    lines = [_format_hit(i, h, f"score={h.get('_score', 0):.4f}") for i, h in enumerate(ranked, 1)]
    return {"result": f"■ 검색 결과 (BM25+Vector, {len(lines)}건)\n\n" + "\n\n".join(lines)}


# ── Graph 조립 ─────────────────────────────────────────────


def _build_graph():
    g = StateGraph(SearchState)
    g.add_node("bm25_search", bm25_search)
    g.add_node("vector_search", vector_search)
    g.add_node("merge_results", merge_results)
    g.add_node("rerank", rerank)

    g.add_edge(START, "bm25_search")
    g.add_edge(START, "vector_search")
    g.add_edge("bm25_search", "merge_results")
    g.add_edge("vector_search", "merge_results")
    g.add_edge("merge_results", "rerank")
    g.add_edge("rerank", END)
    return g.compile()


_graph = _build_graph()


# ── Tool ───────────────────────────────────────────────────


@tool
def search_medical_info(query: str) -> str:
    """증상, 질병명, 치료법 등을 기반으로 의료 문서를 검색합니다.
    키워드 매칭(BM25)과 의미 검색(Vector)을 병렬 수행하여 관련성 높은 결과를 반환합니다.

    Args:
        query: 검색할 의료 키워드 (예: '결핵 치료', '천식 증상', '응급처치')
    """
    return _graph.invoke({"query": query})["result"]
