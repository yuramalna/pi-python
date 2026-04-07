# Build a Research Agent

This tutorial builds a research agent from scratch that can search the web and take notes. It demonstrates the full agent workflow: defining tools, configuring the agent, subscribing to events, and running a multi-turn research task.

## Overview

The research agent has two tools:

- **WebSearchTool** -- Searches the web using the Tavily API
- **NoteTool** -- Saves research notes to memory

The agent loop handles everything automatically: when the LLM decides to search, the agent executes the search tool, feeds the results back, and continues until the research is complete.

## Prerequisites

```bash
pip install pi-llm-agent httpx
```

You will need API keys for:

- An LLM provider (e.g., OpenAI)
- [Tavily](https://tavily.com/) for web search (or substitute your own search API)

## Step 1: Define the WebSearchTool

```python
import httpx
from pi_llm_agent import AgentTool, AgentToolResult
from pi_llm import TextContent


class WebSearchTool(AgentTool):
    """Search the web using the Tavily API."""

    def __init__(self, api_key: str):
        super().__init__(
            name="web_search",
            label="Web Search",
            description="Search the web for current information on a topic",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                },
                "required": ["query"],
            },
        )
        self.api_key = api_key

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        query = params["query"]

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": 5,
                },
            )
            data = response.json()

        # Format results as text
        results = []
        for item in data.get("results", []):
            results.append(
                f"**{item['title']}**\n{item['url']}\n{item.get('content', '')}\n"
            )

        text = f"Search results for: {query}\n\n" + "\n---\n".join(results)
        return AgentToolResult(content=[TextContent(text=text)])
```

## Step 2: Define the NoteTool

```python
class NoteTool(AgentTool):
    """Save a research note."""

    def __init__(self):
        super().__init__(
            name="save_note",
            label="Save Note",
            description="Save a research note with a title and content",
            parameters={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title",
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content",
                    },
                },
                "required": ["title", "content"],
            },
        )
        self.notes: list[dict] = []

    async def execute(self, tool_call_id, params, cancellation=None, on_update=None):
        note = {"title": params["title"], "content": params["content"]}
        self.notes.append(note)
        return AgentToolResult(
            content=[TextContent(text=f"Note saved: {note['title']}")],
            details=note,
        )
```

## Step 3: Create an event printer

Build a subscriber that shows the agent's activity in real time:

```python
from pi_llm_agent import (
    AgentStartEvent,
    AgentEndEvent,
    AgentMessageUpdateEvent,
    ToolExecutionStartEvent,
    ToolExecutionEndEvent,
)
from pi_llm import TextDeltaEvent, ThinkingDeltaEvent


def make_event_printer():
    """Create an event subscriber that prints agent activity."""

    def on_event(event, cancellation):
        if isinstance(event, AgentStartEvent):
            print("=" * 60)
            print("Research agent started")
            print("=" * 60)

        elif isinstance(event, AgentMessageUpdateEvent):
            inner = event.assistant_message_event
            if isinstance(inner, TextDeltaEvent):
                print(inner.delta, end="", flush=True)
            elif isinstance(inner, ThinkingDeltaEvent):
                print(f"[thinking] {inner.delta}", end="", flush=True)

        elif isinstance(event, ToolExecutionStartEvent):
            print(f"\n>>> Calling {event.tool_name}: {event.args}")

        elif isinstance(event, ToolExecutionEndEvent):
            status = "ERROR" if event.is_error else "OK"
            print(f"<<< {event.tool_name}: {status}")

        elif isinstance(event, AgentEndEvent):
            print("\n" + "=" * 60)
            print(f"Research complete. {len(event.messages)} messages generated.")
            print("=" * 60)

    return on_event
```

## Step 4: Wire it all together

```python
import asyncio
import os

from pi_llm_agent import Agent, AgentOptions, InitialAgentState
from pi_llm import get_model
from pi_llm.providers import register_builtin_providers


async def run_research(topic: str):
    register_builtin_providers()

    # Create tools
    note_tool = NoteTool()
    search_tool = WebSearchTool(api_key=os.environ["TAVILY_API_KEY"])

    # Create agent
    agent = Agent(AgentOptions(
        initial_state=InitialAgentState(
            model=get_model("openai", "gpt-4o"),
            system_prompt=(
                "You are a research assistant. When given a topic:\n"
                "1. Search the web for relevant information\n"
                "2. Save key findings as notes\n"
                "3. Provide a summary of your research\n\n"
                "Be thorough -- search multiple angles of the topic."
            ),
            tools=[search_tool, note_tool],
        ),
        get_api_key=lambda provider: os.environ.get("OPENAI_API_KEY"),
    ))

    # Subscribe to events
    agent.subscribe(make_event_printer())

    # Run the research
    await agent.prompt(f"Research the following topic: {topic}")

    # Print collected notes
    print("\n\nCollected Notes:")
    for i, note in enumerate(note_tool.notes, 1):
        print(f"\n{i}. {note['title']}")
        print(f"   {note['content'][:200]}...")

    return agent.state.messages


asyncio.run(run_research("Recent advances in quantum computing"))
```

## How it works

1. `agent.prompt(topic)` adds a `UserMessage` and starts the agent loop.
2. The LLM reads the system prompt and decides to call `web_search`.
3. The agent executes `WebSearchTool.execute()`, producing a `ToolResultMessage`.
4. The tool result is appended to the conversation and the LLM is called again.
5. The LLM may search again, save notes, or produce a final summary.
6. The loop ends when the LLM responds with text only (no more tool calls).
7. Events are emitted at each stage, driving the real-time event printer.

## Adding cancellation support

For long research tasks, add a timeout:

```python
async def run_with_timeout(topic: str, timeout_seconds: int = 60):
    # ... create agent as above ...

    task = asyncio.create_task(agent.prompt(f"Research: {topic}"))

    try:
        await asyncio.wait_for(task, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        agent.abort()
        print("\nResearch timed out, partial results available.")
        await agent.wait_for_idle()
```

## Adding steering

Redirect the agent mid-research:

```python
import time
from pi_llm import UserMessage

async def run_with_steering(topic: str):
    # ... create agent as above ...

    async def steer_after_delay():
        await asyncio.sleep(10)
        agent.steer(UserMessage(
            content="Focus specifically on practical applications, not theory.",
            timestamp=int(time.time() * 1000),
        ))

    # Run both concurrently
    await asyncio.gather(
        agent.prompt(f"Research: {topic}"),
        steer_after_delay(),
    )
```

The steering message is injected between turns, so the LLM sees it before its next response.

## Next steps

- [Agent Lifecycle](../concepts/agent-lifecycle.md) -- Understanding the Agent class
- [Events](../concepts/events.md) -- All 10 event types
- [Hooks](../concepts/hooks.md) -- Intercept tool calls
- [Control Tool Execution](control-tool-execution.md) -- Sequential vs parallel execution
