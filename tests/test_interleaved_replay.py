import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from scripts.benchmark_interleaved_trace import prepare_events, run_batches
from scripts.interleaved_replay import (
    build_manifest,
    load_and_validate_sources,
    reconstruct_prompt,
)


def write_trace(path: Path, trace_id: str, block_lists: list[list[int]]) -> None:
    requests = []
    for index, blocks in enumerate(block_lists):
        requests.append(
            {
                "t": float(index * 10),
                "hash_ids": blocks,
                "input_types": ["tool_result"] if index else ["text"],
                "stop": "tool_use" if index + 1 < len(block_lists) else "end_turn",
            }
        )
    path.write_text(
        json.dumps(
            {
                "id": trace_id,
                "block_size": 64,
                "hash_id_scope": "local",
                "requests": requests,
            }
        )
    )


class InterleavedReplayTest(unittest.TestCase):
    def test_round_robin_batches_preserve_session_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_trace(root / "trace_0001.json", "trace_0001", [[1], [1, 2], [1, 2, 3]])
            write_trace(root / "trace_0002.json", "trace_0002", [[1], [1, 4]])
            write_trace(root / "trace_0003.json", "trace_0003", [[8], [8, 9]])
            manifest = build_manifest(
                root,
                max_traces=3,
                requests_per_trace=3,
                target_prompt_tokens=64,
                max_concurrency=2,
            )

            plan = manifest["dispatch_plan"]
            self.assertEqual(len(plan), 7)
            self.assertEqual(
                [(row["round_index"], row["session_id"]) for row in plan],
                [
                    (0, "trace_0001/main"),
                    (0, "trace_0002/main"),
                    (0, "trace_0003/main"),
                    (1, "trace_0001/main"),
                    (1, "trace_0002/main"),
                    (1, "trace_0003/main"),
                    (2, "trace_0001/main"),
                ],
            )
            for batch_index in {row["batch_index"] for row in plan}:
                batch = [row for row in plan if row["batch_index"] == batch_index]
                self.assertLessEqual(len(batch), 2)
                self.assertEqual(
                    len({row["session_id"] for row in batch}), len(batch)
                )

    def test_local_hash_ids_are_scoped_by_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # The anonymous block IDs are deliberately identical.  They must
            # not map to the same prompt across trace-local hash namespaces.
            write_trace(root / "trace_0001.json", "trace_0001", [[1, 2]])
            write_trace(root / "trace_0002.json", "trace_0002", [[1, 2]])
            manifest = build_manifest(root, target_prompt_tokens=64)
            left = manifest["sessions"][0]["nodes"][0]
            right = manifest["sessions"][1]["nodes"][0]
            self.assertNotEqual(left["prompt_sha256"], right["prompt_sha256"])

    def test_subset_manifest_selection_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_trace(root / "trace_0001.json", "trace_0001", [[1]])
            write_trace(root / "trace_0002.json", "trace_0002", [[2]])
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "traces": [
                            {"relative_path": "trace_0002.json", "selection_rank": 1},
                            {"relative_path": "trace_0001.json", "selection_rank": 2},
                        ]
                    }
                )
            )
            manifest = build_manifest(root, target_prompt_tokens=64)
            self.assertEqual(
                [session["trace_id"] for session in manifest["sessions"]],
                ["trace_0002", "trace_0001"],
            )
            self.assertEqual(
                manifest["config"]["trace_order"],
                "subset_manifest_selection_rank",
            )

    def test_manifest_detects_source_mutation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "trace_0001.json"
            write_trace(path, "trace_0001", [[1], [1, 2]])
            manifest = build_manifest(root, target_prompt_tokens=64)
            trace = json.loads(path.read_text())
            trace["requests"][0]["hash_ids"] = [99]
            path.write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_and_validate_sources(manifest, root)

    def test_reconstructed_prompt_matches_manifest_digest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_trace(root / "trace_0001.json", "trace_0001", [[1], [1, 2]])
            manifest = build_manifest(root, target_prompt_tokens=64)
            sources = load_and_validate_sources(manifest, root)
            session = manifest["sessions"][0]
            for node in session["nodes"]:
                prompt = reconstruct_prompt(
                    session, node, sources[session["session_id"]]
                )
                self.assertEqual(len(prompt), node["prompt_tokens"])

    def test_online_driver_runs_batches_against_mock_server(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_trace(
                root / "trace_0001.json", "trace_0001", [[1], [1, 2], [1, 2, 3]]
            )
            write_trace(
                root / "trace_0002.json", "trace_0002", [[4], [4, 5], [4, 5, 6]]
            )
            manifest = build_manifest(
                root,
                max_traces=2,
                requests_per_trace=3,
                target_prompt_tokens=64,
                max_concurrency=2,
            )
            sources = load_and_validate_sources(manifest, root)
            events, _ = prepare_events(manifest, sources)
            state = {
                "lock": threading.Lock(),
                "inflight": 0,
                "max_inflight": 0,
                "bodies": [],
            }

            def fake_send(_endpoint, _model, prompt, kv_policy, _max_tokens, _timeout):
                body = {"prompt": prompt, "kv_transfer_params": kv_policy}
                started = time.perf_counter()
                with state["lock"]:
                    state["inflight"] += 1
                    state["max_inflight"] = max(
                        state["max_inflight"], state["inflight"]
                    )
                    state["bodies"].append(body)
                time.sleep(0.02)
                with state["lock"]:
                    state["inflight"] -= 1
                completed = time.perf_counter()
                return {
                    "started_monotonic": started,
                    "completed_monotonic": completed,
                    "elapsed_seconds": completed - started,
                    "prompt_tokens_reported": len(prompt),
                    "completion_tokens": 2,
                }

            rows, _ = run_batches(
                manifest,
                events,
                mode="geneakv-r0",
                endpoint="mock://vllm",
                model="mock-model",
                max_tokens=2,
                timeout=1,
                ttl_seconds=600,
                request_sender=fake_send,
            )
            self.assertEqual(len(rows), 6)
            self.assertGreaterEqual(state["max_inflight"], 2)
            measured = [body for body in state["bodies"] if body["kv_transfer_params"]]
            self.assertEqual(len(measured), 6)
            high = [
                body
                for body in measured
                if body["kv_transfer_params"]["geneakv"]["value_class"] == "HIGH"
            ]
            self.assertEqual(len(high), 4)
            for body in high:
                contract = body["kv_transfer_params"]["geneakv"]
                self.assertEqual(contract["protected_prefix_tokens"], len(body["prompt"]))


if __name__ == "__main__":
    unittest.main()
