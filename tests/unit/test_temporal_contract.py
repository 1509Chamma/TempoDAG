import pytest

from tempo_dag.ir.graph import Graph
from tempo_dag.ir_temporal import (
    BufferSpec,
    Edge0,
    EdgeDelta,
    Kernel,
    Process,
    ResetPolicy,
    StateKind,
    StateSpec,
    TemporalStorageKind,
    derive_temporal_execution_contract,
)


def _kernel(kernel_id: str = "kernel") -> Kernel:
    return Kernel(
        kernel_id=kernel_id,
        graph=Graph(values={}, ops={}, graph_inputs=[], graph_outputs=[]),
    )


def test_contract_derives_warmup_from_lag_and_buffer_depth() -> None:
    process = Process(
        process_id="rolling_recurrence",
        kernels={"kernel": _kernel()},
        buffers={
            "window": BufferSpec(
                buffer_id="window",
                dtype="float32",
                shape=(4,),
                depth=8,
            )
        },
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(4,),
            )
        },
        edge0=[Edge0("window", "kernel")],
        edge_delta=[EdgeDelta("kernel", "hidden", lag_cycles=2)],
    )

    contract = derive_temporal_execution_contract(process)

    assert contract.process_id == "rolling_recurrence"
    assert contract.reset_policy == ResetPolicy.ZERO
    assert contract.warmup_timesteps == 7
    assert contract.has_warmup
    assert contract.flush_cycles == 0


def test_contract_uses_metadata_initializer_reset_policy() -> None:
    process = Process(
        process_id="initialized_state",
        kernels={"kernel": _kernel()},
        states={
            "hidden": StateSpec(
                state_id="hidden",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(2,),
                metadata={"initializer": [1.0, 0.0]},
            )
        },
    )

    contract = derive_temporal_execution_contract(process)

    assert contract.reset_policy == ResetPolicy.METADATA_INITIALIZER


def test_contract_maps_short_lag_to_register_and_long_lag_to_shift_register() -> None:
    process = Process(
        process_id="lag_storage",
        kernels={"kernel": _kernel()},
        states={
            "state": StateSpec(
                state_id="state",
                kind=StateKind.HIDDEN,
                dtype="float32",
                shape=(1,),
            )
        },
        edge_delta=[
            EdgeDelta("kernel", "state", lag_cycles=1, value_id="short"),
            EdgeDelta("state", "kernel", lag_cycles=3, value_id="long"),
        ],
    )

    contract = derive_temporal_execution_contract(process)
    storage = {mapping.component_id: mapping for mapping in contract.edge_delta_storage}

    assert storage["kernel->state@1:short"].storage_kind == TemporalStorageKind.REGISTER
    assert (
        storage["state->kernel@3:long"].storage_kind
        == TemporalStorageKind.SHIFT_REGISTER
    )


def test_contract_maps_buffer_depth_to_storage_kind() -> None:
    process = Process(
        process_id="buffer_storage",
        kernels={"kernel": _kernel()},
        buffers={
            "small": BufferSpec("small", dtype="float32", shape=(1,), depth=2),
            "window": BufferSpec("window", dtype="float32", shape=(1,), depth=16),
            "large": BufferSpec("large", dtype="float32", shape=(1,), depth=2048),
        },
    )

    contract = derive_temporal_execution_contract(process)
    storage = {
        mapping.component_id: mapping.storage_kind
        for mapping in contract.buffer_storage
    }

    assert storage == {
        "small": TemporalStorageKind.SHIFT_REGISTER,
        "window": TemporalStorageKind.RING_BUFFER,
        "large": TemporalStorageKind.RAM,
    }


def test_contract_rejects_invalid_flush_cycle_metadata() -> None:
    process = Process(
        process_id="bad_flush",
        kernels={"kernel": _kernel()},
        metadata={"flush_cycles": -1},
    )

    with pytest.raises(ValueError, match="flush_cycles"):
        derive_temporal_execution_contract(process)
