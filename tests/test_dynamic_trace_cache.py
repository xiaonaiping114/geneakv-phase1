import unittest

from scripts.simulate_dynamic_trace_cache import TierCache, online_r1_score


class DynamicTraceCacheTest(unittest.TestCase):
    def test_value_cache_rejects_lower_priority_entry(self):
        cache = TierCache(capacity_blocks=2, eviction="value", ttl_seconds=600)
        cache.insert(("high",), 2, 0, 1.0)
        admitted, rejected = cache.insert(("low",), 2, 1, 0.1)
        self.assertTrue(cache.contains(("high",), 1, touch=False))
        self.assertFalse(cache.contains(("low",), 1, touch=False))
        self.assertEqual((admitted, rejected), (0, 2))

    def test_ttl_removes_entry(self):
        cache = TierCache(capacity_blocks=10, eviction="lru", ttl_seconds=5)
        cache.insert(("x",), 2, 0, 0)
        self.assertTrue(cache.contains(("x",), 4))
        self.assertFalse(cache.contains(("x",), 6))

    def test_r1_score_uses_only_online_arguments(self):
        score = online_r1_score(100, ["tool_result"], 0.9, 10)
        self.assertGreater(score, online_r1_score(100, ["text"], 0.2, 1000))


if __name__ == "__main__":
    unittest.main()
