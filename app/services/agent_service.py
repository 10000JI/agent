import asyncio
from datetime import datetime
import json
import os
import traceback
import uuid

from app.utils.logger import log_execution, custom_logger

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

_RECURSION_LIMIT = 15


def _configure_opik():
    from app.core.config import settings

    if settings.OPIK is None:
        return
    opik_settings = settings.OPIK
    if opik_settings.URL_OVERRIDE:
        os.environ["OPIK_URL_OVERRIDE"] = opik_settings.URL_OVERRIDE
    if opik_settings.API_KEY:
        os.environ["OPIK_API_KEY"] = opik_settings.API_KEY
    if opik_settings.WORKSPACE:
        os.environ["OPIK_WORKSPACE"] = opik_settings.WORKSPACE
    if opik_settings.PROJECT:
        os.environ["OPIK_PROJECT_NAME"] = opik_settings.PROJECT


_configure_opik()


class AgentService:
    def __init__(self):
        from langchain_openai import ChatOpenAI
        from app.core.config import settings

        self.model = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            streaming=True,
        )

        self.opik_tracer = None
        if settings.OPIK is not None:
            from opik.integrations.langchain import OpikTracer
            self.opik_tracer = OpikTracer(
                tags=["medical-agent"],
                metadata={"model": settings.OPENAI_MODEL},
            )

        self.checkpointer = None
        self.agent = None

    async def _init_checkpointer(self):
        if self.checkpointer is not None:
            return
        from langgraph.checkpoint.memory import MemorySaver
        self.checkpointer = MemorySaver()

    def _create_agent(self):
        from app.agents.medical_agent import create_medical_agent
        assert self.checkpointer is not None
        self.agent = create_medical_agent(
            model=self.model,
            checkpointer=self.checkpointer,
        )
        if self.opik_tracer is not None:
            from opik.integrations.langchain import track_langgraph
            self.agent = track_langgraph(self.agent, self.opik_tracer)

    # -----------------------------------------------------------------------
    # SSE 이벤트 빌더
    # -----------------------------------------------------------------------

    @staticmethod
    def _done_event(content: str, metadata: dict | None = None,
                    message_id: str | None = None, error: str | None = None) -> str:
        payload = {
            "step": "done",
            "message_id": message_id or str(uuid.uuid4()),
            "role": "assistant",
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        if error is not None:
            payload["error"] = error
        return json.dumps(payload, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # 청크 파싱
    # -----------------------------------------------------------------------

    def _parse_chunk(self, chunk: dict):
        """astream chunk를 SSE 이벤트로 변환. (중간 이벤트 리스트, done 이벤트 or None) 반환."""
        events = []
        done_event = None

        for step, event in chunk.items():
            if not event or step not in ("model", "tools"):
                continue
            messages = event.get("messages", [])
            if not messages:
                continue
            message = messages[0]

            if step == "model":
                structured = event.get("structured_response")
                if structured:
                    done_event = self._done_event(
                        content=getattr(structured, "content", ""),
                        metadata=self._handle_metadata(getattr(structured, "metadata", None)),
                        message_id=getattr(structured, "message_id", None),
                    )
                    continue

                tool_calls = message.tool_calls
                if not tool_calls:
                    continue

                first_tool = tool_calls[0]
                if first_tool.get("name") == "ChatResponse":
                    args = first_tool.get("args", {})
                    done_event = self._done_event(
                        content=args.get("content", ""),
                        metadata=self._handle_metadata(args.get("metadata")),
                        message_id=args.get("message_id"),
                    )
                else:
                    events.append(json.dumps({
                        "step": "model",
                        "tool_calls": [tc["name"] for tc in tool_calls],
                    }))

            elif step == "tools":
                if message.name == "ChatResponse":
                    continue
                events.append(json.dumps({
                    "step": "tools",
                    "name": message.name,
                    "content": message.content,
                }, ensure_ascii=False))

        return events, done_event

    # -----------------------------------------------------------------------
    # 실제 대화 로직 — astream(progress) + fallback(done)
    # -----------------------------------------------------------------------

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        await self._init_checkpointer()
        self._create_agent()
        custom_logger.info(f"사용자 메시지: {user_messages}")

        config = {
            "configurable": {"thread_id": str(thread_id)},
            "recursion_limit": _RECURSION_LIMIT,
        }

        done_sent = False

        # Phase 1: astream — 중간 이벤트(model, tools) 전송
        try:
            async for chunk in self.agent.astream(
                {"messages": [HumanMessage(content=user_messages)]},
                config=config,
                stream_mode="updates",
            ):
                events, done_event = self._parse_chunk(chunk)
                for ev in events:
                    yield ev
                if done_event:
                    yield done_event
                    done_sent = True
        except GraphRecursionError:
            custom_logger.warning(f"GraphRecursionError: recursion_limit={_RECURSION_LIMIT}")
        except Exception as e:
            custom_logger.error(f"Stream error: {e}")
            custom_logger.error(traceback.format_exc())

        # Phase 2: done 미전송 시 — LLM 직접 호출 fallback
        if not done_sent:
            try:
                await self._patch_pending_tool_calls(config)

                state = await self.agent.aget_state(config)
                messages = list(state.values.get("messages", []))

                pending = self._find_pending_tool_calls(messages)
                for tc in pending:
                    messages.append(ToolMessage(
                        content="[도구 호출 횟수 제한으로 실행되지 않았습니다]",
                        tool_call_id=tc["id"],
                    ))

                messages.append(SystemMessage(
                    content="도구 호출 횟수 제한에 도달했습니다. "
                            "지금까지 수집된 정보를 바탕으로 사용자에게 최선의 답변을 생성하세요. "
                            "추가 도구 호출 없이 텍스트로만 답변하세요."
                ))
                response = await self.model.ainvoke(messages)
                yield self._done_event(content=response.content)
            except Exception as fallback_err:
                custom_logger.error(f"Fallback 실패: {fallback_err}")
                yield self._done_event(content="처리 중 오류가 발생했습니다. 다시 시도해주세요.")

    # -----------------------------------------------------------------------
    # 유틸리티
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_pending_tool_calls(messages: list) -> list[dict]:
        answered_ids: set[str] = {
            msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
        }
        pending: list[dict] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc["id"] not in answered_ids:
                        pending.append(tc)
        return pending

    async def _patch_pending_tool_calls(self, config: dict) -> None:
        try:
            state = await self.agent.aget_state(config)
            messages = list(state.values.get("messages", []))
            pending = self._find_pending_tool_calls(messages)
            if not pending:
                return
            custom_logger.info(f"미완료 tool_call {len(pending)}건 체크포인트 패치")
            patch_messages = [
                ToolMessage(content="[오류로 인해 실행되지 않았습니다]", tool_call_id=tc["id"])
                for tc in pending
            ]
            await self.agent.aupdate_state(
                config, {"messages": patch_messages}, as_node="tools",
            )
        except Exception as patch_err:
            custom_logger.error(f"체크포인트 패치 실패: {patch_err}")

    @staticmethod
    def _handle_metadata(metadata) -> dict:
        if not metadata:
            return {}
        if isinstance(metadata, dict):
            return metadata
        if hasattr(metadata, "model_dump"):
            return metadata.model_dump()
        return {}
