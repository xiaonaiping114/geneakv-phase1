import unittest

from geneakv.lineage import LineageGraph
from geneakv.model import AgentEvent, EventType, LineagePolicy, ValueClass


def event(seq, node, kind, parent=None, tokens=1025, branch="main"):
    return AgentEvent(
        session_id="s1",
        node_id=node,
        parent_id=parent,
        event_type=kind,
        event_seq=seq,
        epoch=1,
        prefix_tokens=tokens,
        branch_id=branch,
    )


class LineageGraphTest(unittest.TestCase):
    def test_tool_call_is_high_and_uncapped(self):
        graph = LineageGraph(block_size=128)
        policy = graph.apply(event(1, "n1", EventType.TOOL_CALL))
        self.assertEqual(policy.value_class, ValueClass.HIGH)
        self.assertIsNone(policy.protected_prefix_tokens)
        self.assertIn("TOOL_RESUME_EXPECTED", policy.reason_codes)

    def test_shared_ancestor_becomes_high(self):
        graph = LineageGraph()
        graph.apply(event(1, "root", EventType.REQUEST_START))
        graph.apply(event(2, "a", EventType.BRANCH_FORK, "root", branch="a"))
        graph.apply(event(3, "b", EventType.BRANCH_FORK, "root", branch="b"))
        self.assertEqual(
            graph.policy_for(graph.nodes[("s1", "root")]).value_class,
            ValueClass.HIGH,
        )

    def test_abort_is_low(self):
        graph = LineageGraph()
        graph.apply(event(1, "root", EventType.REQUEST_START))
        policy = graph.apply(event(2, "dead", EventType.BRANCH_ABORT, "root"))
        self.assertEqual(policy.value_class, ValueClass.LOW)
        self.assertEqual(policy.protected_prefix_tokens, 0)

    def test_contract_round_trip(self):
        graph = LineageGraph()
        policy = graph.apply(event(1, "n1", EventType.TOOL_CALL))
        parsed = LineagePolicy.from_kv_transfer_params(policy.to_kv_transfer_params())
        self.assertEqual(parsed, policy)

    def test_missing_boundary_means_full_prefix(self):
        params = {
            "geneakv": {
                "schema_version": 1,
                "session_id": "s1",
                "node_id": "n1",
                "event_seq": 1,
                "epoch": 1,
                "value_class": "HIGH",
                "reason_codes": ["TOOL_RESUME_EXPECTED"],
            }
        }
        parsed = LineagePolicy.from_kv_transfer_params(params)
        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed.protected_prefix_tokens)

    def test_explicit_boundary_is_preserved(self):
        params = {
            "geneakv": {
                "schema_version": 1,
                "session_id": "s1",
                "node_id": "n1",
                "event_seq": 1,
                "epoch": 1,
                "value_class": "HIGH",
                "protected_prefix_tokens": 2048,
                "reason_codes": [],
            }
        }
        parsed = LineagePolicy.from_kv_transfer_params(params)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.protected_prefix_tokens, 2048)

    def test_trace_aliases_and_parent_key(self):
        raw = {
            "session_id": "s2",
            "node_id": "child",
            "parent_node_id": "root",
            "event_type": "ABORT",
            "event_seq": 7,
            "prefix_tokens": 512,
            "branch_id": "B",
            "state": "ABORTED",
        }
        parsed = AgentEvent.from_dict(raw)
        self.assertEqual(parsed.event_type, EventType.BRANCH_ABORT)
        self.assertEqual(parsed.parent_id, "root")
        self.assertEqual(parsed.metadata["state"], "ABORTED")


if __name__ == "__main__":
    unittest.main()
