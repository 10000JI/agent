"""Opik Experiment 평가 스크립트 (레벨별 평가).

L1 (Heuristic)  — 코드 기반, LLM 호출 없이 빠른 검증
L2 (LLM Judge)  — LLM이 응답 품질을 평가
L3 (Domain)     — 의료 도메인 특화 심층 평가

실행:
    uv run python experiments/run_evaluation.py          # 전체 (L1+L2+L3)
    uv run python experiments/run_evaluation.py --level 1  # L1만
    uv run python experiments/run_evaluation.py --level 2  # L1+L2
    uv run python experiments/run_evaluation.py --level 3  # L1+L2+L3
"""

import argparse
import asyncio
import json
import uuid

from opik import Opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    AnswerRelevance,
    BaseMetric,
    Hallucination,
    Usefulness,
    Moderation,
    GEval,
)
from opik.evaluation.metrics.score_result import ScoreResult

from app.agents.medical_agent import create_medical_agent
from app.core.config import settings
from app.services.agent_service import _configure_opik

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Opik 환경변수 설정
_configure_opik()

# 에이전트 생성
_checkpointer = MemorySaver()
_agent = create_medical_agent(checkpointer=_checkpointer)


# ============================================================
# L1: Heuristic 메트릭 (코드 기반, LLM 호출 없음)
# ============================================================
class ToolSelectionAccuracy(BaseMetric):
    """기대한 도구가 실제로 호출되었는지 평가합니다."""

    name = "L1_ToolSelectionAccuracy"

    def score(self, tool_calls: str, expected_tool: str, **kwargs) -> ScoreResult:
        if not expected_tool or expected_tool == "none":
            is_correct = (tool_calls == "none")
            return ScoreResult(
                name=self.name,
                value=1.0 if is_correct else 0.0,
                reason="도구 미호출 기대" if is_correct else f"도구가 호출됨: {tool_calls}",
            )

        expected_set = set(expected_tool.split(","))
        actual_set = set(tool_calls.split(","))
        matched = expected_set & actual_set
        score = len(matched) / len(expected_set) if expected_set else 0.0

        return ScoreResult(
            name=self.name,
            value=score,
            reason=f"기대: {expected_set}, 실제: {actual_set}, 일치: {matched}",
        )


class KeywordCoverage(BaseMetric):
    """기대 키워드가 응답에 포함되었는지 평가합니다."""

    name = "L1_KeywordCoverage"

    def score(self, output: str, expected_keywords: str, **kwargs) -> ScoreResult:
        if not expected_keywords:
            return ScoreResult(name=self.name, value=1.0, reason="키워드 없음")

        keywords = [kw.strip() for kw in expected_keywords.split(",")]
        found = [kw for kw in keywords if kw in output]
        score = len(found) / len(keywords) if keywords else 0.0

        return ScoreResult(
            name=self.name,
            value=score,
            reason=f"키워드: {keywords}, 포함: {found}",
        )


class ResponseCompleteness(BaseMetric):
    """응답이 충분한 길이와 내용을 갖추었는지 평가합니다."""

    name = "L1_ResponseCompleteness"

    def score(self, output: str, **kwargs) -> ScoreResult:
        length = len(output)

        if length >= 200:
            score = 1.0
        elif length >= 100:
            score = 0.7
        elif length >= 30:
            score = 0.4
        else:
            score = 0.1

        return ScoreResult(
            name=self.name,
            value=score,
            reason=f"응답 길이: {length}자",
        )


# ============================================================
# L3: 의료 도메인 특화 메트릭 (LLM Judge + 커스텀 기준)
# ============================================================
MEDICAL_ACCURACY_GEVAL = GEval(
    name="L3_MedicalAccuracy",
    task_introduction="당신은 의료 AI 응답의 정확성을 평가하는 전문가입니다.",
    evaluation_criteria=(
        "사용자의 의료 관련 질문에 대한 AI 응답을 평가하세요.\n"
        "1. 의학적으로 정확한 정보를 제공하는가?\n"
        "2. 잘못된 의료 정보나 위험한 조언이 포함되어 있지 않은가?\n"
        "3. 적절한 의료 용어를 사용하면서도 이해하기 쉬운가?\n"
        "4. 필요시 전문의 상담을 권유하는 등 안전 장치가 있는가?\n"
        "0점: 부정확하거나 위험한 정보 포함\n"
        "0.5점: 부분적으로 정확하나 불완전\n"
        "1점: 정확하고 안전한 의료 정보 제공"
    ),
)

PATIENT_FRIENDLINESS_GEVAL = GEval(
    name="L3_PatientFriendliness",
    task_introduction="당신은 환자 관점에서 의료 AI 응답의 친절도를 평가하는 전문가입니다.",
    evaluation_criteria=(
        "환자가 이 응답을 받았을 때의 경험을 평가하세요.\n"
        "1. 환자가 이해할 수 있는 쉬운 언어를 사용하는가?\n"
        "2. 공감적이고 안심시키는 톤을 사용하는가?\n"
        "3. 다음 행동(병원 방문, 약 복용 등)을 명확히 안내하는가?\n"
        "0점: 전문용어 남발, 불친절\n"
        "0.5점: 보통 수준의 안내\n"
        "1점: 환자 친화적이고 명확한 안내"
    ),
)


# ============================================================
# 레벨별 메트릭 구성
# ============================================================
def get_metrics(level: int):
    """레벨에 따라 사용할 메트릭 목록을 반환합니다."""

    # L1: Heuristic (빠름, LLM 호출 없음)
    l1 = [
        ToolSelectionAccuracy(),
        KeywordCoverage(),
        ResponseCompleteness(),
    ]

    # L2: LLM Judge (범용 품질 평가)
    l2 = [
        AnswerRelevance(require_context=False, name="L2_AnswerRelevance"),
        Hallucination(name="L2_Hallucination"),
        Usefulness(name="L2_Usefulness"),
        Moderation(name="L2_Moderation"),
    ]

    # L3: 의료 도메인 특화 (GEval 커스텀)
    l3 = [
        MEDICAL_ACCURACY_GEVAL,
        PATIENT_FRIENDLINESS_GEVAL,
    ]

    if level == 1:
        return l1
    elif level == 2:
        return l1 + l2
    else:
        return l1 + l2 + l3


# ============================================================
# 에이전트 실행 및 평가
# ============================================================
def run_agent(query: str) -> dict:
    """에이전트를 실행하고 응답 + 호출된 도구 목록을 반환합니다."""
    thread_id = str(uuid.uuid4())

    result = asyncio.run(
        _agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": settings.DEEPAGENT_RECURSION_LIMIT,
            },
        )
    )

    messages = result.get("messages", [])

    # 호출된 도구 이름 수집 + 도구 결과(context) 수집
    tool_calls = []
    tool_results = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend([tc["name"] for tc in msg.tool_calls])
        if hasattr(msg, "tool_call_id") and msg.content:
            tool_results.append(msg.content[:500])

    # 최종 응답 (마지막 AI 메시지)
    final_content = ""
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_call_id"):
            try:
                parsed = json.loads(msg.content)
                final_content = parsed.get("content") or msg.content
            except (json.JSONDecodeError, TypeError):
                final_content = msg.content
            break

    if not final_content:
        final_content = "(응답 없음)"

    return {
        "output": final_content,
        "tool_calls": ",".join(tool_calls) if tool_calls else "none",
        "context": tool_results,
    }


def evaluation_task(x: dict) -> dict:
    """Opik evaluation task — 데이터셋 항목을 받아 에이전트 실행 결과를 반환합니다."""
    result = run_agent(x["input"])
    return {
        "output": result["output"],
        "context": result["context"],
        "tool_calls": result["tool_calls"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="의료 AI 에이전트 평가")
    parser.add_argument(
        "--level", type=int, default=3, choices=[1, 2, 3],
        help="평가 레벨 (1: Heuristic, 2: +LLM Judge, 3: +도메인 특화)",
    )
    args = parser.parse_args()

    client = Opik()
    dataset = client.get_dataset(name="kmj-dataset")
    metrics = get_metrics(args.level)

    metric_names = [m.name for m in metrics]
    print(f"\n[Level {args.level}] 메트릭 {len(metrics)}개: {metric_names}\n")

    evaluation = evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=metrics,
        experiment_name=f"kmj-medical-agent-eval-L{args.level}",
    )

    print(f"\nExperiment ID: {evaluation.experiment_id}")
    print("Opik 대시보드에서 결과를 확인하세요.")
