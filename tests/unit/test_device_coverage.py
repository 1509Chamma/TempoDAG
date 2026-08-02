"""Coverage for device board validation branches and registry loading."""

import pytest

from tempo_dag.device import DeviceRegistry, FPGADevice, Memory, Resources


@pytest.fixture
def device():
    return FPGADevice(
        name="cov_device",
        vendor="CovVendor",
        part_number="COV-001",
        resources=Resources(luts=100000, ffs=200000, dsps=500, bram_36k=100),
        memory=Memory(on_chip_kb=4096, external_bandwidth_gbps=19.2),
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: setattr(d, "vendor", ""), "vendor.*non-empty"),
        (lambda d: setattr(d, "part_number", ""), "part_number.*non-empty"),
        (lambda d: setattr(d.resources, "ffs", 0), "ffs.*positive"),
        (lambda d: setattr(d.resources, "dsps", -1), "dsps.*non-negative"),
        (lambda d: setattr(d.resources, "bram_36k", -1), "bram_36k.*non-negative"),
        (lambda d: setattr(d.resources, "bram_18k", -1), "bram_18k.*non-negative"),
        (lambda d: setattr(d.memory, "on_chip_kb", 0), "on_chip_kb.*positive"),
        (
            lambda d: setattr(d.memory, "external_latency_ns", -1.0),
            "external_latency_ns.*non-negative",
        ),
        (lambda d: setattr(d.io, "pcie_lanes", -1), "pcie_lanes.*non-negative"),
        (
            lambda d: setattr(d.policies, "max_clock_mhz", 0.0),
            "max_clock_mhz.*positive",
        ),
        (
            lambda d: setattr(d.policies, "target_clock_mhz", 0.0),
            "target_clock_mhz.*positive",
        ),
    ],
)
def test_validate_rejects_invalid_field(device, mutate, match):
    mutate(device)
    with pytest.raises(ValueError, match=match):
        device.validate()


def test_registry_default_config_dir_loads_repo_presets():
    registry = DeviceRegistry()
    presets = registry.list_presets()
    assert presets, "expected bundled device presets under configs/devices"
    first = registry.get_preset(presets[0])
    assert first["name"] == presets[0]


def test_registry_rejects_malformed_preset_json(tmp_path):
    (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Failed to load preset"):
        DeviceRegistry(config_dir=str(tmp_path))
