import json
import tempfile
from pathlib import Path

import pytest

from tempo_dag.device import DeviceRegistry, FPGADevice


class TestDeviceRegistryIntegration:
    @pytest.fixture
    def temp_config_dir(self):
        # Create temporary config directory with test presets
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            preset1 = {
                "name": "test_device_1",
                "vendor": "TestVendor",
                "part_number": "TEST-001",
                "resources": {
                    "luts": 100000,
                    "ffs": 200000,
                    "dsps": 500,
                    "bram_36k": 100,
                },
                "memory": {
                    "on_chip_kb": 4096,
                    "external_bandwidth_gbps": 19.2,
                },
            }

            preset2 = {
                "name": "test_device_2",
                "vendor": "AnotherVendor",
                "part_number": "TEST-002",
                "resources": {
                    "luts": 200000,
                    "ffs": 400000,
                    "dsps": 1000,
                    "bram_36k": 200,
                },
                "memory": {
                    "on_chip_kb": 8192,
                    "external_bandwidth_gbps": 38.4,
                },
            }

            with open(config_dir / "test_device_1.json", "w") as f:
                json.dump(preset1, f)

            with open(config_dir / "test_device_2.json", "w") as f:
                json.dump(preset2, f)

            yield config_dir

    # Load multiple devices end-to-end test
    def test_load_multiple_devices(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))

        device1 = registry.load_device("test_device_1")
        device2 = registry.load_device("test_device_2")

        assert device1.resources.luts == 100000
        assert device2.resources.luts == 200000
        assert device1.vendor != device2.vendor

    # Integration test with project presets
    def test_registry_with_project_presets(self):
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs" / "devices"

        if not config_dir.exists():
            pytest.skip("Project preset files not available")

        registry = DeviceRegistry(str(config_dir))
        presets = registry.list_presets()

        assert len(presets) > 0

        for preset_name in presets:
            device = registry.load_device(preset_name)
            assert isinstance(device, FPGADevice)
            device.validate()

    # Intel Stratix 10 MX preset test
    def test_intel_s10mx_preset_values(self):
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs" / "devices"

        if not config_dir.exists():
            pytest.skip("Project preset files not available")

        registry = DeviceRegistry(str(config_dir))

        if "intel_s10mx" not in registry.list_presets():
            pytest.skip("intel_s10mx preset not available")

        device = registry.load_device("intel_s10mx")

        assert device.vendor == "Intel"
        assert device.part_number == "1SM21CHU2F53E1VG"
        assert device.resources.luts == 2073000
        assert device.resources.dsps == 7920
        assert device.memory.external_bandwidth_gbps == 4096.0
        assert device.io.pcie_lanes == 16
        assert device.io.pcie_gen == 3
        assert device.capabilities.supports_fp64 is False
        assert device.policies.max_clock_mhz == 1000.0
        assert device.policies.target_clock_mhz <= device.policies.max_clock_mhz

    # Kria KV260 preset test
    def test_xilinx_kv260_preset_values(self):
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs" / "devices"

        if not config_dir.exists():
            pytest.skip("Project preset files not available")

        registry = DeviceRegistry(str(config_dir))

        if "xilinx_kv260" not in registry.list_presets():
            pytest.skip("xilinx_kv260 preset not available")

        device = registry.load_device("xilinx_kv260")

        assert device.vendor == "Xilinx"
        assert device.part_number == "xck26-sfvc784-2LV"
        assert device.resources.luts == 256000
        assert device.resources.dsps == 1200
        assert device.resources.bram_36k == 144
        assert device.memory.on_chip_kb == 2952
        assert device.memory.external_bandwidth_gbps == 153.6
        assert device.io.pcie_lanes == 0
        assert device.io.pcie_gen == 3
        assert device.io.other_interfaces["qspi"] == "512Mb"
        assert device.io.other_interfaces["usb"] == "4x USB 3.0/2.0"

    # Alveo U250 preset test
    def test_xilinx_u250_preset_values(self):
        project_root = Path(__file__).parent.parent.parent
        config_dir = project_root / "configs" / "devices"

        if not config_dir.exists():
            pytest.skip("Project preset files not available")

        registry = DeviceRegistry(str(config_dir))

        if "xilinx_u250" not in registry.list_presets():
            pytest.skip("xilinx_u250 preset not available")

        device = registry.load_device("xilinx_u250")

        assert device.vendor == "Xilinx"
        assert device.part_number == "xcu250-figd2104-2L"
        assert device.resources.luts == 1728000
        assert device.resources.ffs == 3456000
        assert device.resources.dsps == 12288
        assert device.memory.on_chip_kb == 55296
        assert device.memory.external_bandwidth_gbps == 616.0
        assert device.io.pcie_lanes == 16
        assert device.io.pcie_gen == 3
        assert device.io.other_interfaces["ddr4"] == "64GB ECC @ 2400 MT/s"
        assert device.io.other_interfaces["qsfp28"] == "2x 100GbE"
        assert device.policies.power_budget_w == 225.0
