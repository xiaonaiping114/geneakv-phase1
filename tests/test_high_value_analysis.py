import unittest

from scripts.analyze_high_value_traces import iter_streams, lcp_blocks


class HighValueAnalysisTest(unittest.TestCase):
    def test_lcp_blocks_stops_at_first_difference(self):
        self.assertEqual(lcp_blocks([1, 2, 3, 4], [1, 2, 9, 4]), 2)

    def test_main_and_nested_streams_are_separated(self):
        trace = {
            "id": "t1",
            "block_size": 64,
            "requests": [
                {"t": 0, "hash_ids": [1, 2]},
                {
                    "t": 10,
                    "type": "subagent",
                    "agent_id": "a1",
                    "requests": [{"t": 0, "hash_ids": [1, 3]}],
                },
                {"t": 20, "hash_ids": [1, 2, 4]},
            ],
        }
        streams = list(iter_streams(trace))
        self.assertEqual(len(streams), 2)
        nested = next(stream for stream in streams if stream.kind == "subagent")
        main = next(stream for stream in streams if stream.kind == "main")
        self.assertEqual(nested.start_time, 10)
        self.assertEqual(len(nested.requests), 1)
        self.assertEqual(len(main.requests), 2)


if __name__ == "__main__":
    unittest.main()
