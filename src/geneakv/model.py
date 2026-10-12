"""Stable data contracts shared by the trace parser and vLLM connector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    REQUEST_START = "REQUEST_START"
    MODEL_RESPONSE = "MODEL_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    BRANCH_FORK = "BRANCH_FORK"
    BRANCH_COMMIT = "BRANCH_COMMIT"
    BRANCH_ABORT = "BRANCH_ABORT"
    RETRY = "RETRY"
    SESSION_END = "SESSION_END"


class ValueClass(str, Enum):
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"


@dataclass(frozen=True)
class AgentEvent:
    session_id: str
    node_id: str
    parent_id: str | None
    event_type: EventType
    event_seq: int
    epoch: int
    prefix_tokens: int
    branch_id: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AgentEvent":
        event_aliases = {
            "REQUEST_END": EventType.MODEL_RESPONSE,
            "RESUME": EventType.MODEL_RESPONSE,
            "FORK": EventType.BRANCH_FORK,
            "BRANCH_STEP": EventType.REQUEST_START,
            "COMMIT": EventType.BRANCH_COMMIT,
            "ABORT": EventType.BRANCH_ABORT,
            "ERROR": EventType.MODEL_RESPONSE,
            "EXPIRE": EventType.SESSION_END,
        }
        raw_event_type = str(raw["event_type"]).upper()
        event_type = event_aliases.get(raw_event_type)
        if event_type is None:
            event_type = EventType(raw_event_type)
        known_fields = {
            "session_id",
            "node_id",
            "parent_id",
            "parent_node_id",
            "event_type",
            "event_seq",
            "epoch",
            "prefix_tokens",
            "branch_id",
            "metadata",
        }
        metadata = dict(raw.get("metadata", {}))
        metadata.update({k: v for k, v in raw.items() if k not in known_fields})
        return cls(
            session_id=str(raw["session_id"]),
            node_id=str(raw["node_id"]),
            parent_id=(
                str(raw.get("parent_id") or raw.get("parent_node_id"))
                if raw.get("parent_id") or raw.get("parent_node_id")
                else None
            ),
            event_type=event_type,
            event_seq=int(raw["event_seq"]),
            epoch=int(raw.get("epoch", 0)),
            prefix_tokens=max(0, int(raw.get("prefix_tokens", 0))),
            branch_id=str(raw.get("branch_id", "main")),
            metadata=metadata,
        )


@dataclass
class LineageNode:
    session_id: str
    node_id: str
    parent_id: str | None
    branch_id: str
    prefix_tokens: int
    epoch: int
    last_event_seq: int
    last_event_type: EventType
    children: set[str] = field(default_factory=set)
    live_children: set[str] = field(default_factory=set)
    waiting_for_tool: bool = False
    committed: bool = False
    terminal: bool = False
    retry_count: int = 0


@dataclass(frozen=True)
class LineagePolicy:
    schema_version: int
    session_id: str
    node_id: str
    event_seq: int
    epoch: int
    value_class: ValueClass
    # None means "all currently confirmed, cacheable prefix blocks".  A
    # non-negative integer is retained only for explicit boundary experiments.
    protected_prefix_tokens: int | None
    reason_codes: tuple[str, ...]
    ttl_seconds: int = 300

    def to_kv_transfer_params(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["value_class"] = self.value_class.value
        payload["reason_codes"] = list(self.reason_codes)
        return {"geneakv": payload}

    @classmethod
    def from_kv_transfer_params(
        cls, params: dict[str, Any] | None
    ) -> "LineagePolicy | None":
        if not isinstance(params, dict):
            return None
        raw = params.get("geneakv")
        if not isinstance(raw, dict) or int(raw.get("schema_version", 0)) != 1:
            return None
        try:
            raw_protected = raw.get("protected_prefix_tokens")
            protected_prefix_tokens = (
                None
                if raw_protected is None
                else max(0, int(raw_protected))
            )
            return cls(
                schema_version=1,
                session_id=str(raw["session_id"]),
                node_id=str(raw["node_id"]),
                event_seq=int(raw["event_seq"]),
                epoch=int(raw["epoch"]),
                value_class=ValueClass(str(raw["value_class"]).upper()),
                protected_prefix_tokens=protected_prefix_tokens,
                reason_codes=tuple(str(v) for v in raw.get("reason_codes", [])),
                ttl_seconds=max(1, int(raw.get("ttl_seconds", 300))),
            )
        except (KeyError, TypeError, ValueError):
            return None
