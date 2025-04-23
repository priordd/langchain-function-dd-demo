# ref.: https://python.langchain.com/docs/how_to/tool_calling/

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core import tools


load_dotenv()


from ddtrace.llmobs import LLMObs
from ddtrace.llmobs import decorators
from ddtrace.llmobs.decorators import embedding, llm, retrieval, workflow, tool

LLMObs.enable(
    api_key=os.getenv("DD_API_KEY"),
    site=os.getenv("DD_SITE"),
    ml_app=os.getenv("DD_LLMOBS_ML_APP"),
    agentless_enabled=os.getenv("DD_LLMOBS_AGENTLESS_ENABLED"),
    env=os.getenv("DD_ENV"),
    service=os.getenv("DD_SERVICE"),
)


@tools.tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tools.tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


@workflow
def main(query):
    llm = init_chat_model("gpt-4o-mini", name="invoke_llm", model_provider="openai")
    tools = [add, multiply]
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)

    response = llm_with_tools.invoke(messages)
    print(response.content)
    return response.content


if __name__ == "__main__":
    query = "What is 3 * 12? Also, what is 11 + 49?"
    main(query)
