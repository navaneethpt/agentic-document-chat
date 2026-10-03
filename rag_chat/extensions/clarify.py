"""Example optional agent that requests a missing detail from the user."""

from pydantic import Field

from rag_chat.chat import Answer
from rag_chat.workflows import NodeConfig, NodeType, register_node


class ClarifyConfig(NodeConfig):
    question: str = Field(default="Which document should I use?", min_length=1)


def build_clarify(client, on_event, config):
    def run(state):
        question = config.get("question", "Which document should I use?")
        if on_event:
            on_event({"event": "clarify", "question": question})
        return {"answer": Answer(question, [])}, None

    return run


register_node(NodeType(
    key="clarify", label="Ask for clarification", kind="agent",
    description="Ask the user for a missing detail.", outputs=(),
    config_model=ClarifyConfig, factory=build_clarify,
    provides=("answer",),
))
