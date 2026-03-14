import httpx
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

from app.utils.logger import custom_logger


@wrap_tool_call
async def handle_tool_errors(request, handler):
    """모든 도구의 예외를 사용자 행동 기준으로 분류하여 처리한다.

    create_agent 내부의 ToolNode는 기본적으로 도구 예외를 다시 raise하므로,
    이 middleware에서 예외를 잡아 ToolMessage로 변환하여 대화가 깨지지 않도록 한다.

    분류 기준:
        1. 타임아웃 → 재시도 유도
        2. 네트워크/연결 실패 → 인프라 문제 안내
        3. 기타 (파싱 에러, ES 에러 등) → 일반 오류 안내
    """
    tool_name = request.tool_call["name"]
    try:
        return await handler(request)
    except httpx.TimeoutException:
        custom_logger.error(f"{tool_name}: API 타임아웃")
        return ToolMessage(
            content="외부 API 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.",
            tool_call_id=request.tool_call["id"],
        )
    except httpx.NetworkError:
        custom_logger.error(f"{tool_name}: 네트워크 연결 실패")
        return ToolMessage(
            content="외부 서비스에 연결할 수 없습니다. 네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요.",
            tool_call_id=request.tool_call["id"],
        )
    except Exception as e:
        custom_logger.error(f"{tool_name} 오류: {e}")
        return ToolMessage(
            content=f"도구 실행 중 오류가 발생했습니다: {str(e)}",
            tool_call_id=request.tool_call["id"],
        )
