import unittest

from scripts.evaluate_r1_high_value import enrich_online_features, evaluate


class R1HighValueTest(unittest.TestCase):
    def test_current_future_does_not_change_current_score(self):
        base = {
            "trace_id": "t",
            "stream_id": "t/main",
            "stream_kind": "main",
            "request_index": 0,
            "input_types": ["text"],
            "current_blocks": 100,
            "r0_high": True,
            "next_delay_seconds": 10,
            "next_lcp_blocks": 90,
            "next_lcp_ratio": 0.9,
        }
        changed = dict(base, next_delay_seconds=9999, next_lcp_blocks=0, next_lcp_ratio=0)
        self.assertEqual(
            enrich_online_features([base])[0]["r1_score"],
            enrich_online_features([changed])[0]["r1_score"],
        )

    def test_previous_transition_can_affect_later_score(self):
        rows = [
            {
                "trace_id": "t",
                "stream_id": "t/main",
                "stream_kind": "main",
                "request_index": 0,
                "input_types": ["text"],
                "current_blocks": 100,
                "r0_high": True,
                "next_delay_seconds": 10,
                "next_lcp_blocks": 100,
                "next_lcp_ratio": 1.0,
            },
            {
                "trace_id": "t",
                "stream_id": "t/main",
                "stream_kind": "main",
                "request_index": 1,
                "input_types": ["tool_result"],
                "current_blocks": 110,
                "r0_high": True,
                "next_delay_seconds": 20,
                "next_lcp_blocks": 100,
                "next_lcp_ratio": 0.9,
            },
        ]
        result = enrich_online_features(rows)
        self.assertTrue(result[1]["r1_tool_chain"])
        self.assertEqual(result[1]["r1_history_reuse_ema"], 1.0)
        self.assertEqual(result[1]["r1_history_delay_ema"], 10.0)

    def test_evaluate_outputs_all_policies(self):
        rows = []
        for index in range(10):
            rows.append(
                {
                    "trace_id": "t",
                    "stream_id": "t/main",
                    "stream_kind": "main",
                    "request_index": index,
                    "input_types": ["tool_result"] if index else ["text"],
                    "current_blocks": 10 + index,
                    "r0_high": True,
                    "next_delay_seconds": 10,
                    "next_lcp_blocks": 9 + index,
                    "next_lcp_ratio": 0.9,
                }
            )
        summary = evaluate(rows, 600, (0.5,), 2)
        self.assertEqual(
            {row["policy"] for row in summary["results"]},
            {"R0-random", "Size-only", "R1-online", "Oracle"},
        )


if __name__ == "__main__":
    unittest.main()
