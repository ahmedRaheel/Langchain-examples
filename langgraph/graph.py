from state import AgentState
from langgraph.graph import (
    StateGraph,
    START,
    END
)

from planner import planner
from executor import execute_action
from runner import format_answer

def create_planner_node(tools):

    def planner_node(
        state: AgentState
    ):

        action = planner(
            state,
            tools
        )

        print("\nPlanner Selected:")
        print(action)

        return {
            "action": action,
            "finished": action == "FINISH"
        }

    return planner_node


def create_tool_node(client):

    async def tool_node(
        state: AgentState
    ):

        action = state["action"]

        observation = await execute_action(
            action,
            state,
            client
        )

        print("\nTool Observation:")
        print(observation)

        return {
            "observations": (
                state["observations"]
                + [
                    {
                        "step": state["current_step"],
                        "action": action,
                        "observation": observation
                    }
                ]
            ),
            "actions": (
                state["actions"]
                + [action]
            ),
            "current_step": (
                state["current_step"] + 1
            )
        }

    return tool_node


def answer_node(
    state: AgentState
):

    answer = format_answer(
        state
    )

    print("\nFinal Answer:")
    print(answer)

    return {
        "final_answer": answer,
        "finished": True
    }


def route_after_planner(
    state: AgentState
):

    if state["action"] == "FINISH":
        return "finish"

    return "tool"


def route_after_tool(
    state: AgentState
):

    if (
        state["current_step"]
        >= state["max_steps"]
    ):
        return "stop"

    return "continue"


def build_graph(
    tools,
    client
):

    builder = StateGraph(
        AgentState
    )

    builder.add_node(
        "planner",
        create_planner_node(tools)
    )

    builder.add_node(
        "tool",
        create_tool_node(client)
    )

    builder.add_node(
        "answer",
        answer_node
    )

    builder.add_edge(
        START,
        "planner"
    )

    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "tool": "tool",
            "finish": "answer"
        }
    )

    builder.add_conditional_edges(
        "tool",
        route_after_tool,
        {
            "continue": "planner",
            "stop": "answer"
        }
    )

    builder.add_edge(
        "answer",
        END
    )

    return builder.compile()
