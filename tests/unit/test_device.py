import pytest

from tempo_dag.device import (
    IO,
    Capabilities,
    FPGADevice,
    Memory,
    Policies,
    Resources,
)


class TestResources:
    # Basic resource test
    def test_create_basic_resources(self):
        res = Resources(luts=100000, ffs=200000, dsps=500, bram_36k=100)
        assert res.luts == 100000
        assert res.ffs == 200000
        assert res.dsps == 500
        assert res.bram_36k == 100
        assert res.bram_18k == 0

    # Resource with 18k BRAM test
    def test_create_resources_with_18k(self):
        res = Resources(
            luts=100000,
            ffs=200000,
            dsps=500,
            bram_36k=100,
            bram_18k=50,
        )
        assert res.bram_18k == 50


class TestMemory:
    # Basic memory test
    def test_create_basic_memory(self):
        mem = Memory(on_chip_kb=4096, external_bandwidth_gbps=19.2)
        assert mem.on_chip_kb == 4096
        assert mem.external_bandwidth_gbps == 19.2
        assert mem.external_latency_ns == 100.0

    # Memory with custom latency test
    def test_create_memory_with_latency(self):
        mem = Memory(
            on_chip_kb=4096,
            external_bandwidth_gbps=19.2,
            external_latency_ns=50.0,
        )
        assert mem.external_latency_ns == 50.0


class TestIO:
    # Default IO configuration test
    def test_create_default_io(self):
        io = IO()
        assert io.pcie_lanes == 16
        assert io.pcie_gen == 4
        assert io.other_interfaces == {}

    # IO with custom interfaces test
    def test_create_io_with_interfaces(self):
        io = IO(
            pcie_lanes=8,
            pcie_gen=3,
            other_interfaces={"ethernet": "100Gbps", "usb": "3.1"},
        )
        assert io.pcie_lanes == 8
        assert io.pcie_gen == 3
        assert io.other_interfaces["ethernet"] == "100Gbps"


class TestCapabilities:
    # Default capabilities test
    def test_create_default_capabilities(self):
        cap = Capabilities()
        assert cap.supports_fp32 is True
        assert cap.supports_fp64 is False
        assert cap.supports_int8 is True

    # Custom capabilities test
    def test_create_custom_capabilities(self):
        cap = Capabilities(
            supports_fp32=True,
            supports_fp64=True,
            supports_bfloat16=True,
        )
        assert cap.supports_fp64 is True
        assert cap.supports_bfloat16 is True


class TestPolicies:
    # Default policies test
    def test_create_default_policies(self):
        p = Policies()
        assert p.max_clock_mhz == 250.0
        assert p.target_clock_mhz == 200.0
        assert p.default_precision == "int8"
        assert p.power_budget_w is None

    # Custom policies test
    def test_create_custom_policies(self):
        p = Policies(
            max_clock_mhz=300.0,
            target_clock_mhz=250.0,
            power_budget_w=75.0,
        )
        assert p.max_clock_mhz == 300.0
        assert p.power_budget_w == 75.0


class TestFPGADevice:
    @pytest.fixture
    def basic_device(self):
        # Create a basic FPGADevice for testing
        return FPGADevice(
            name="test_device",
            vendor="TestVendor",
            part_number="TEST-001",
            resources=Resources(luts=100000, ffs=200000, dsps=500, bram_36k=100),
            memory=Memory(on_chip_kb=4096, external_bandwidth_gbps=19.2),
        )

    # Basic device creation test
    def test_create_basic_device(self, basic_device):
        assert basic_device.name == "test_device"
        assert basic_device.vendor == "TestVendor"
        assert basic_device.part_number == "TEST-001"

    # Device to dictionary conversion test
    def test_device_to_dict(self, basic_device):
        d = basic_device.to_dict()
        assert d["name"] == "test_device"
        assert d["vendor"] == "TestVendor"
        assert d["resources"]["luts"] == 100000
        assert d["memory"]["on_chip_kb"] == 4096

    # Device from dictionary creation test
    def test_device_from_dict(self, basic_device):
        d = basic_device.to_dict()
        device2 = FPGADevice.from_dict(d)

        assert device2.name == basic_device.name
        assert device2.vendor == basic_device.vendor
        assert device2.resources.luts == basic_device.resources.luts

    # Merge scalar overrides test
    def test_merge_overrides_scalars(self, basic_device):
        overrides = {
            "name": "modified_device",
            "policies": {
                "target_clock_mhz": 300.0,
            },
        }

        modified = basic_device.merge_overrides(overrides)

        assert modified.name == "modified_device"
        assert modified.policies.target_clock_mhz == 300.0
        assert basic_device.name == "test_device"
        assert basic_device.policies.target_clock_mhz == 200.0

    # Merge nested dictionary overrides test
    def test_merge_overrides_nested_dict(self, basic_device):
        overrides = {
            "policies": {
                "max_clock_mhz": 280.0,
                "power_budget_w": 50.0,
            }
        }

        modified = basic_device.merge_overrides(overrides)

        assert modified.policies.max_clock_mhz == 280.0
        assert modified.policies.power_budget_w == 50.0
        assert modified.policies.default_precision == "int8"

    # Deep merge of nested structures test
    def test_merge_overrides_deep_merge(self, basic_device):
        overrides = {
            "resources": {
                "dsps": 1000,
            },
            "io": {
                "pcie_lanes": 8,
            },
        }

        modified = basic_device.merge_overrides(overrides)

        assert modified.resources.dsps == 1000
        assert modified.io.pcie_lanes == 8
        assert modified.resources.luts == basic_device.resources.luts
        assert modified.io.pcie_gen == 4

    # Validation missing name test
    def test_validate_missing_name(self, basic_device):
        basic_device.name = ""
        with pytest.raises(ValueError, match="name.*non-empty"):
            basic_device.validate()

    # Validation negative LUTs test
    def test_validate_negative_luts(self, basic_device):
        basic_device.resources.luts = -100
        with pytest.raises(ValueError, match="luts.*positive"):
            basic_device.validate()

    # Validation negative bandwidth test
    def test_validate_negative_external_bandwidth(self, basic_device):
        basic_device.memory.external_bandwidth_gbps = -10.0
        with pytest.raises(ValueError, match="external_bandwidth_gbps.*non-negative"):
            basic_device.validate()

    # Validation invalid PCIe generation test
    def test_validate_invalid_pcie_gen(self, basic_device):
        basic_device.io.pcie_gen = 2
        with pytest.raises(ValueError, match="pcie_gen.*must be 3, 4, or 5"):
            basic_device.validate()

    # Validation target clock exceeds max test
    def test_validate_target_exceeds_max_clock(self, basic_device):
        basic_device.policies.target_clock_mhz = 300.0
        basic_device.policies.max_clock_mhz = 250.0
        with pytest.raises(ValueError, match="target_clock_mhz.*cannot exceed"):
            basic_device.validate()

    # Validation negative power budget test
    def test_validate_negative_power_budget(self, basic_device):
        basic_device.policies.power_budget_w = -10.0
        with pytest.raises(ValueError, match="power_budget_w.*positive"):
            basic_device.validate()

