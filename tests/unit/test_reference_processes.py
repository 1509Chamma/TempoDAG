from tempo_dag.examples import (
    delay_line_process,
    initialized_state_process,
    recurrent_state_process,
    rolling_window_process,
    temporal_reference_processes,
)
from tempo_dag.ir_temporal import (
    ResetPolicy,
    TemporalStorageKind,
    derive_temporal_execution_contract,
)


def test_reference_processes_validate() -> None:
    processes = temporal_reference_processes()

    assert {process.process_id for process in processes} == {
        "reference_delay_line",
        "reference_recurrent_state",
        "reference_rolling_window",
        "reference_initialized_state",
    }
    for process in processes:
        process.validate()


def test_delay_line_reference_exercises_positive_lag_storage() -> None:
    contract = derive_temporal_execution_contract(delay_line_process())

    assert contract.warmup_timesteps == 3
    assert contract.edge_delta_storage[0].storage_kind == (
        TemporalStorageKind.SHIFT_REGISTER
    )


def test_recurrent_state_reference_exercises_register_feedback() -> None:
    contract = derive_temporal_execution_contract(recurrent_state_process())

    assert contract.warmup_timesteps == 1
    assert contract.edge_delta_storage[0].storage_kind == TemporalStorageKind.REGISTER


def test_rolling_window_reference_exercises_buffer_warmup() -> None:
    contract = derive_temporal_execution_contract(rolling_window_process())

    assert contract.warmup_timesteps == 7
    assert contract.buffer_storage[0].storage_kind == TemporalStorageKind.RING_BUFFER


def test_initialized_state_reference_exercises_reset_policy() -> None:
    contract = derive_temporal_execution_contract(initialized_state_process())

    assert contract.reset_policy == ResetPolicy.METADATA_INITIALIZER
