import asyncio
import contextlib
from datetime import datetime
import json
import uuid

from app.utils.logger import log_execution, custom_logger
from app.agents.medical_agent import create_medical_agent
from app.core.config import settings

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError


# ============================================================
# 체크포인터 설정 (대화 기록 유지)
# 1안: MemorySaver (서버 재시작 시 초기화)
# 2안 (추후): SqliteSaver로 교체 시 아래 주석 해제
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# _checkpointer = AsyncSqliteSaver.from_conn_string("chat_history.db")
# ============================================================
_checkpointer = MemorySaver()


class AgentService:
    def __init__(self):
        self.agent = None
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    def _create_agent(self, thread_id: uuid.UUID = None):
        """의료 전문 LangChain 에이전트 생성"""
        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            streaming=True,
        )
        self.agent = create_medical_agent(
            llm=llm,
            checkpointer=_checkpointer,
        )

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """사용자 메시지를 처리하고 스트리밍 응답을 생성합니다."""
        try:
            self._create_agent(thread_id=thread_id)
            custom_logger.info(f"사용자 메시지: {user_messages}")

            agent_stream = self.agent.astream(
                {"messages": [HumanMessage(content=user_messages)]},
                config={"configurable": {"thread_id": str(thread_id)}},
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.create_task(agent_iterator.__anext__())
            progress_task = asyncio.create_task(self.progress_queue.get())

            while True:
                pending = {agent_task}
                if progress_task is not None:
                    pending.add(progress_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # Progress 이벤트 처리
                if progress_task in done:
                    try:
                        progress_event = progress_task.result()
                        yield json.dumps(progress_event, ensure_ascii=False)
                        progress_task = asyncio.create_task(self.progress_queue.get())
                    except asyncio.CancelledError:
                        progress_task = None
                    except Exception as e:
                        custom_logger.error(f"Error in progress_task: {e}")
                        progress_task = None

                # 에이전트 스트림 처리
                if agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        agent_task = None
                        break
                    except Exception as e:
                        custom_logger.error(f"Error in agent_task: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        agent_task = None
                        yield json.dumps(self._error_response(str(e)), ensure_ascii=False)
                        break

                    custom_logger.info(f"에이전트 청크: {chunk}")
                    try:
                        for step, event in chunk.items():
                            if not event or step not in ("agent", "model", "tools"):
                                continue
                            messages = event.get("messages", [])
                            if not messages:
                                continue
                            message = messages[0]

                            # agent/model 단계: 도구 호출 또는 최종 응답
                            if step in ("agent", "model"):
                                tool_calls = message.tool_calls
                                if tool_calls:
                                    # 도구 호출 중 → "model" 이벤트
                                    tool_names = [t["name"] for t in tool_calls]
                                    yield json.dumps({"step": "model", "tool_calls": tool_names}, ensure_ascii=False)
                                elif message.content:
                                    # 도구 호출 없이 최종 응답 → "done" 이벤트
                                    yield json.dumps({
                                        "step": "done",
                                        "message_id": str(uuid.uuid4()),
                                        "role": "assistant",
                                        "content": message.content,
                                        "metadata": {},
                                        "created_at": datetime.utcnow().isoformat(),
                                    }, ensure_ascii=False)

                            # tools 단계: 도구 실행 결과
                            elif step == "tools":
                                try:
                                    content_val = json.loads(message.content)
                                except (json.JSONDecodeError, TypeError):
                                    content_val = message.content
                                yield json.dumps({
                                    "step": "tools",
                                    "name": message.name,
                                    "content": content_val,
                                }, ensure_ascii=False)

                    except Exception as e:
                        custom_logger.error(f"Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        yield json.dumps(self._error_response(str(e)), ensure_ascii=False)
                        break

                    agent_task = asyncio.create_task(agent_iterator.__anext__())

            # 남은 progress 이벤트 정리
            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            while not self.progress_queue.empty():
                try:
                    remaining = self.progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                yield json.dumps(remaining, ensure_ascii=False)

        except Exception as e:
            import traceback
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(traceback.format_exc())
            yield json.dumps(
                self._error_response(str(e) if not isinstance(e, GraphRecursionError) else None),
                ensure_ascii=False,
            )

    @staticmethod
    def _error_response(error: str = None) -> dict:
        """에러 응답 포맷 생성"""
        return {
            "step": "done",
            "message_id": str(uuid.uuid4()),
            "role": "assistant",
            "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
            "metadata": {},
            "created_at": datetime.utcnow().isoformat(),
            "error": error,
        }
