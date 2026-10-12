"""Incremental Agent lineage DAG and deterministic phase-1 policy rules."""

from __future__ import annotations

from collections import defaultdict

from .model import AgentEvent, EventType, LineageNode, LineagePolicy, ValueClass


class LineageGraph:
    """Incrementally builds a per-session DAG from ordered Agent events.

    Phase 1 deliberately uses transparent rules instead of a trained model:
    tool-wait nodes and shared live ancestors are HIGH; aborted/session-ended
    nodes are LOW; all other nodes are NORMAL.
    """

    def __init__(self, block_size: int = 128, ttl_seconds: int = 300):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self.ttl_seconds = ttl_seconds
        self.nodes: dict[tuple[str, str], LineageNode] = {}
        self.last_seq: dict[str, int] = defaultdict(lambda: -1)

    def apply(self, event: AgentEvent) -> LineagePolicy:
        if event.event_seq <= self.last_seq[event.session_id]:
            node = self.nodes[(event.session_id, event.node_id)]
            return self.policy_for(node)

        key = (event.session_id, event.node_id)
        node = self.nodes.get(key)
        if node is None:
            node = LineageNode(
                session_id=event.session_id,
                node_id=event.node_id,
                parent_id=event.parent_id,
                branch_id=event.branch_id,
                prefix_tokens=event.prefix_tokens,
                epoch=event.epoch,
                last_event_seq=event.event_seq,
                last_event_type=event.event_type,
            )
            self.nodes[key] = node
        else:
            node.prefix_tokens = max(node.prefix_tokens, event.prefix_tokens)
            node.epoch = max(node.epoch, event.epoch)
            node.last_event_seq = event.event_seq
            node.last_event_type = event.event_type

        parent = None
        if event.parent_id:
            parent = self.nodes.get((event.session_id, event.parent_id))
            if parent is not None:
                parent.children.add(event.node_id)
                parent.live_children.add(event.branch_id)

        if event.event_type == EventType.TOOL_CALL:
            node.waiting_for_tool = True
        elif event.event_type == EventType.TOOL_RESULT:
            node.waiting_for_tool = False
            if parent is not None:
                parent.waiting_for_tool = False
        elif event.event_type == EventType.BRANCH_FORK:
            node.live_children.update(
                str(branch_id)
                for branch_id in event.metadata.get("child_branch_ids", [])
            )
        elif event.event_type == EventType.BRANCH_COMMIT:
            node.committed = True
            if parent is not None:
                parent.live_children = {event.branch_id}
        elif event.event_type == EventType.BRANCH_ABORT:
            node.terminal = True
            if parent is not None:
                parent.live_children.discard(event.branch_id)
        elif event.event_type == EventType.RETRY:
            node.retry_count += 1
            if parent is not None:
                parent.retry_count += 1
        elif event.event_type == EventType.SESSION_END:
            node.terminal = True

        self.last_seq[event.session_id] = event.event_seq
        return self.policy_for(node)

    def policy_for(self, node: LineageNode) -> LineagePolicy:
        reasons: list[str] = []
        if node.terminal:
            value_class = ValueClass.LOW
            reasons.append("TERMINAL_OR_ABORTED")
        elif node.waiting_for_tool:
            value_class = ValueClass.HIGH
            reasons.append("TOOL_RESUME_EXPECTED")
        elif len(node.live_children) >= 2:
            value_class = ValueClass.HIGH
            reasons.append("SHARED_LIVE_ANCESTOR")
        elif node.committed:
            value_class = ValueClass.HIGH
            reasons.append("COMMITTED_CONTINUATION")
        elif node.retry_count > 0:
            value_class = ValueClass.HIGH
            reasons.append("RETRY_SHARED_PREFIX")
        else:
            value_class = ValueClass.NORMAL
            reasons.append("NO_STRONG_REUSE_SIGNAL")

        # HIGH is admitted without an artificial per-request token cap.  The
        # connector resolves None to all confirmed full blocks at store time.
        protected = None if value_class == ValueClass.HIGH else 0

        return LineagePolicy(
            schema_version=1,
            session_id=node.session_id,
            node_id=node.node_id,
            event_seq=node.last_event_seq,
            epoch=node.epoch,
            value_class=value_class,
            protected_prefix_tokens=protected,
            reason_codes=tuple(reasons),
            ttl_seconds=self.ttl_seconds,
        )
