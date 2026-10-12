"""External vLLM connector implementing GeneaKV selective CPU admission.

This module intentionally lives outside the vLLM/vLLM-Ascend repositories.
It reuses Ascend's NPU DMA worker and changes only scheduler-side admission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm_ascend.distributed.kv_transfer.kv_pool.simple_cpu_offload.simple_cpu_offload_connector import (
    AscendSimpleCPUOffloadConnector,
)

from .model import LineagePolicy, ValueClass

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)


class GeneaKVOffloadScheduler(SimpleCPUOffloadScheduler):
    """Eager CPU offload with request policy and prefix-boundary admission."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._audited_skip_reqs: set[str] = set()
        self._audited_lookup_reqs: set[str] = set()
        cpu_bytes = sum(t.size for t in self.cpu_kv_cache_config.kv_cache_tensors)
        self._bytes_per_block = cpu_bytes // self.num_cpu_blocks

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Audit HBM/CPU/recompute token sources for every new request."""
        cpu_hit_tokens, is_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if request.request_id not in self._audited_lookup_reqs:
            cpu_tokens = int(cpu_hit_tokens or 0)
            reusable_tokens = max(0, request.num_prompt_tokens - 1)
            hbm_tokens = min(max(0, num_computed_tokens), reusable_tokens)
            recompute_tokens = max(0, reusable_tokens - hbm_tokens - cpu_tokens)
            policy = LineagePolicy.from_kv_transfer_params(request.kv_transfer_params)
            value_class = policy.value_class.value if policy else "NONE"
            logger.warning(
                "GENEAKV_LOOKUP request=%s class=%s prefix_tokens=%d "
                "hbm_hit_tokens=%d cpu_hit_tokens=%d recompute_tokens=%d",
                request.request_id,
                value_class,
                reusable_tokens,
                hbm_tokens,
                cpu_tokens,
                recompute_tokens,
            )
            self._audited_lookup_reqs.add(request.request_id)
        return cpu_hit_tokens, is_async

    def _prepare_eager_store_specs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[list[int], list[int], list[str]]:
        merged_gpu_block_ids: list[int] = []
        merged_cpu_block_ids: list[int] = []
        req_ids: list[str] = []

        gpu_block_pool = self._gpu_block_pool
        if gpu_block_pool is None:
            return [], [], []
        cpu_block_pool = self.cpu_block_pool
        num_free = cpu_block_pool.get_num_free_blocks()
        kv_cache_groups = self.cpu_kv_cache_config.kv_cache_groups
        num_groups = len(kv_cache_groups)
        gpu_blocks_this_step: set[int] = set()

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            state = self._reqs_to_store.get(req_id)
            if state is None or state.finished:
                continue

            if preempted:
                state.block_ids = tuple([] for _ in range(num_groups))
                state.num_stored_blocks = [0] * num_groups
            if new_block_id_groups:
                for group_idx in range(min(num_groups, len(new_block_id_groups))):
                    new_ids = new_block_id_groups[group_idx]
                    if new_ids is not None:
                        state.block_ids[group_idx].extend(new_ids)

            if scheduler_output.num_scheduled_tokens.get(req_id, 0) == 0:
                continue

            req = state.request
            policy = LineagePolicy.from_kv_transfer_params(req.kv_transfer_params)
            if policy is None:
                logger.debug("GENEAKV_STORE_SKIP request=%s reason=NO_POLICY", req_id)
                continue
            if policy.value_class != ValueClass.HIGH:
                if req_id not in self._audited_skip_reqs:
                    logger.warning(
                        "GENEAKV_STORE_SKIP request=%s class=%s reasons=%s",
                        req_id,
                        policy.value_class.value,
                        ",".join(policy.reason_codes),
                    )
                    self._audited_skip_reqs.add(req_id)
                continue

            confirmed_tokens = req.num_computed_tokens - req.num_output_placeholders
            admitted_tokens = (
                confirmed_tokens
                if policy.protected_prefix_tokens is None
                else min(confirmed_tokens, policy.protected_prefix_tokens)
            )
            if admitted_tokens <= 0:
                continue

            gpu_block_ids: list[int] = []
            block_hashes_to_store: list[bytes] = []
            advanced_per_group = [0] * num_groups
            out_of_space = False

            for group_idx in range(num_groups):
                already_stored = state.num_stored_blocks[group_idx]
                group_gpu_ids = state.block_ids[group_idx]
                group_block_size = kv_cache_groups[group_idx].kv_cache_spec.block_size
                ready_blocks = admitted_tokens // group_block_size
                scannable = group_gpu_ids[already_stored:ready_blocks]

                for gpu_block_id in scannable:
                    gpu_block = gpu_block_pool.blocks[gpu_block_id]
                    if gpu_block.is_null:
                        advanced_per_group[group_idx] += 1
                        continue
                    block_hash = gpu_block.block_hash
                    if block_hash is None:
                        break
                    if (
                        gpu_block_id in gpu_blocks_this_step
                        or cpu_block_pool.cached_block_hash_to_block.get_one_block(
                            block_hash
                        )
                        is not None
                    ):
                        advanced_per_group[group_idx] += 1
                        continue
                    if num_free <= 0:
                        out_of_space = True
                        break
                    num_free -= 1
                    gpu_block_ids.append(gpu_block_id)
                    block_hashes_to_store.append(block_hash)
                    advanced_per_group[group_idx] += 1
                if out_of_space:
                    break

            if gpu_block_ids:
                cpu_blocks = cpu_block_pool.get_new_blocks(len(gpu_block_ids))
                cpu_block_ids = [block.block_id for block in cpu_blocks]
                for cpu_block, block_hash in zip(cpu_blocks, block_hashes_to_store):
                    cpu_block._block_hash = block_hash

                req_ids.append(req_id)
                merged_gpu_block_ids.extend(gpu_block_ids)
                merged_cpu_block_ids.extend(cpu_block_ids)
                gpu_blocks_this_step.update(gpu_block_ids)
                gpu_block_pool.touch(
                    [gpu_block_pool.blocks[block_id] for block_id in gpu_block_ids]
                )
                logger.warning(
                    "GENEAKV_STORE_ADMIT request=%s session=%s node=%s "
                    "boundary_tokens=%s stored_tokens=%d blocks=%d bytes=%d "
                    "reasons=%s",
                    req_id,
                    policy.session_id,
                    policy.node_id,
                    (
                        "FULL"
                        if policy.protected_prefix_tokens is None
                        else str(policy.protected_prefix_tokens)
                    ),
                    len(cpu_block_ids) * self.block_size,
                    len(cpu_block_ids),
                    len(cpu_block_ids) * self._bytes_per_block,
                    ",".join(policy.reason_codes),
                )

            for group_idx in range(num_groups):
                state.num_stored_blocks[group_idx] += advanced_per_group[group_idx]

        return merged_gpu_block_ids, merged_cpu_block_ids, req_ids


class GeneaKVConnector(AscendSimpleCPUOffloadConnector):
    """Ascend connector with native NPU worker + GeneaKV scheduler policy."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if role != KVConnectorRole.SCHEDULER or self.scheduler_manager is None:
            return

        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        if bool(extra.get("lazy_offload", False)):
            raise ValueError("GeneaKV phase-1 requires eager selective admission")
        world_size = vllm_config.parallel_config.world_size
        cpu_capacity = int(extra.get("cpu_bytes_to_use", DEFAULT_CPU_CAPACITY_BYTES))
        cpu_capacity_per_rank = int(
            extra.get("cpu_bytes_to_use_per_rank", cpu_capacity // world_size)
        )
        self.scheduler_manager = GeneaKVOffloadScheduler(
            vllm_config,
            kv_cache_config,
            cpu_capacity_per_rank,
            lazy_offload=False,
        )
        logger.warning(
            "GeneaKVConnector enabled: selective HIGH-value prefix admission, "
            "per_rank=%.2f GB",
            cpu_capacity_per_rank / (1024**3),
        )
