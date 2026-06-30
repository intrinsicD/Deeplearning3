"""Explicit agent graph definitions.

The graph is intentionally small and auditable. It describes which node types a
runtime may execute; actual model/tool behavior is supplied by the runtime via
handlers/tools. No side effects are performed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Mapping


class NodeType(str, Enum):
    OBSERVE = "OBSERVE"
    ENCODE = "ENCODE"
    THINK = "THINK"
    ROUTE = "ROUTE"
    TOOL_CALL = "TOOL_CALL"
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    DECODE = "DECODE"
    KB_READ = "KB_READ"
    KB_WRITE = "KB_WRITE"


@dataclass(frozen=True)
class AgentNode:
    """A named graph node with an explicit type."""

    node_type: NodeType
    name: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.node_type.value


@dataclass
class AgentGraph:
    """Declarative set of allowed agent node types and route-action mapping."""

    nodes: List[AgentNode] = field(default_factory=list)
    action_nodes: Dict[str, NodeType] = field(default_factory=dict)
    start: NodeType = NodeType.OBSERVE

    def __post_init__(self) -> None:
        if not self.nodes:
            self.nodes = [AgentNode(node_type=t) for t in NodeType]
        self._allowed = {node.node_type for node in self.nodes}
        if self.start not in self._allowed:
            raise ValueError(f"Graph start node {self.start.value} is not in graph nodes")
        for action, node_type in self.action_nodes.items():
            if node_type not in self._allowed:
                raise ValueError(f"Action {action!r} maps to unavailable node {node_type.value}")

    @property
    def allowed_node_types(self) -> set[NodeType]:
        return set(self._allowed)

    def allows(self, node_type: NodeType) -> bool:
        return node_type in self._allowed

    def node_for_action(self, action: str) -> NodeType:
        """Resolve a router action to an explicit node type."""
        if action in self.action_nodes:
            return self.action_nodes[action]
        try:
            node_type = NodeType(action)
        except ValueError as exc:
            raise KeyError(f"No graph node mapping for route action {action!r}") from exc
        if not self.allows(node_type):
            raise KeyError(f"Route action {action!r} resolves to unavailable node {node_type.value}")
        return node_type

    @classmethod
    def with_nodes(
        cls,
        node_types: Iterable[NodeType],
        *,
        action_nodes: Dict[str, NodeType] | None = None,
        start: NodeType = NodeType.OBSERVE,
    ) -> "AgentGraph":
        return cls(
            nodes=[AgentNode(node_type=t) for t in node_types],
            action_nodes={} if action_nodes is None else dict(action_nodes),
            start=start,
        )


def default_agent_graph(tool_actions: Iterable[str] | None = None) -> AgentGraph:
    """Return the default explicit agent graph.

    ``tool_actions`` are extra action strings (e.g. a registry's tool dispatch
    ids from :meth:`ExpertRegistry.tool_actions`) that should resolve to a
    ``TOOL_CALL`` node, so a router selecting a tool expert routes to a runnable
    tool rather than the generic ``"TOOL_CALL"`` action.
    """
    action_nodes = {
        "STOP": NodeType.ROUTE,
        "THINK": NodeType.THINK,
        "TOOL_CALL": NodeType.TOOL_CALL,
        "MEMORY_READ": NodeType.MEMORY_READ,
        "MEMORY_WRITE": NodeType.MEMORY_WRITE,
        "DECODE": NodeType.DECODE,
        "KB_READ": NodeType.KB_READ,
        "KB_WRITE": NodeType.KB_WRITE,
    }
    for action in tool_actions or ():
        action_nodes[action] = NodeType.TOOL_CALL
    return AgentGraph(action_nodes=action_nodes)


__all__ = ["AgentGraph", "AgentNode", "NodeType", "default_agent_graph"]

