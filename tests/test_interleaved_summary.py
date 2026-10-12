import unittest

from scripts.summarize_interleaved_matrix import comparisons, render_html


class InterleavedSummaryTest(unittest.TestCase):
    def test_same_capacity_comparison(self):
        rows = [
            {
                "mode": "eager",
                "capacity_gib": 8,
                "requests": 10,
                "high_store_candidates": 5,
                "high_resume_requests": 4,
                "effective_hit_rate": 0.20,
                "local_hits": 100.0,
                "external_hits": 20.0,
                "external_restore_coverage_proxy": 0.10,
                "benchmark_elapsed_seconds": 10.0,
                "request_mean_ms": 90.0,
                "request_p95_ms": 140.0,
                "resume_mean_ms": 100.0,
                "resume_p50_ms": 80.0,
                "resume_p95_ms": 150.0,
                "resume_p99_ms": 170.0,
                "throughput_rps": 2.0,
                "source_file": "eager-cpu8g.json",
            },
            {
                "mode": "geneakv-r0",
                "capacity_gib": 8,
                "requests": 10,
                "high_store_candidates": 5,
                "high_resume_requests": 4,
                "effective_hit_rate": 0.25,
                "local_hits": 100.0,
                "external_hits": 50.0,
                "external_restore_coverage_proxy": 0.25,
                "benchmark_elapsed_seconds": 9.0,
                "request_mean_ms": 75.0,
                "request_p95_ms": 110.0,
                "resume_mean_ms": 80.0,
                "resume_p50_ms": 60.0,
                "resume_p95_ms": 120.0,
                "resume_p99_ms": 140.0,
                "throughput_rps": 2.2,
                "source_file": "geneakv-r0-cpu8g.json",
            },
        ]
        delta = comparisons(rows)[0]
        self.assertAlmostEqual(delta["hit_rate_lift_pp"], 5.0)
        self.assertAlmostEqual(delta["hit_rate_relative_gain_percent"], 25.0)
        self.assertAlmostEqual(delta["resume_mean_reduction_percent"], 20.0)
        self.assertAlmostEqual(delta["resume_p50_reduction_percent"], 25.0)
        self.assertEqual(delta["external_hit_delta"], 30.0)
        document = render_html(rows, [delta])
        self.assertIn("GeneaKV-R0", document)
        self.assertIn("+5.000 pp", document)
        self.assertIn("HIGH Resume 延迟", document)


if __name__ == "__main__":
    unittest.main()
