import json
import tempfile
from pathlib import Path

import pytest

from tempo_dag.device import DeviceRegistry, FPGADevice


class TestDeviceRegistry:
    @pytest.fixture
    def temp_config_dir(self):
        # Create temporary config directory with test presets
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create sample preset in JSON format
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

            # Write presets as JSON
            with open(config_dir / "test_device_1.json", "w") as f:
                json.dump(preset1, f)

            with open(config_dir / "test_device_2.json", "w") as f:
                json.dump(preset2, f)

            yield config_dir

    # Registry initialization test
    def test_registry_initialization(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))
        assert len(registry._presets) == 2

    # List presets test
    def test_list_presets(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))
        presets = registry.list_presets()

        assert "test_device_1" in presets
        assert "test_device_2" in presets
        assert len(presets) == 2
        assert presets == sorted(presets)

    # Get preset by name test
    def test_get_preset(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))
        preset = registry.get_preset("test_device_1")

        assert preset["name"] == "test_device_1"
        assert preset["vendor"] == "TestVendor"
        assert preset["resources"]["luts"] == 100000

    # Get preset not found test
    def test_get_preset_not_found(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))

        with pytest.raises(KeyError, match="not found"):
            registry.get_preset("nonexistent_device")

    # Load device from preset test
    def test_load_device_from_preset(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))
        device = registry.load_device("test_device_1")

        assert isinstance(device, FPGADevice)
        assert device.name == "test_device_1"
        assert device.vendor == "TestVendor"
        assert device.resources.luts == 100000

    # Load device with overrides test
    def test_load_device_with_overrides(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))

        overrides = {
            "name": "test_device_1_customized",
            "policies": {
                "target_clock_mhz": 200.0,
            },
        }

        device = registry.load_device("test_device_1", overrides=overrides)

        assert device.name == "test_device_1_customized"
        assert device.policies.target_clock_mhz == 200.0
        assert device.vendor == "TestVendor"

    # Nested overrides test
    def test_load_device_with_nested_overrides(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))

        # Override that would be invalid
        overrides = {
            "policies": {
                "target_clock_mhz": 500.0,  # Exceeds default max of 250
                "max_clock_mhz": 600.0,  # Override max to allow higher target
            }
        }

        device = registry.load_device("test_device_1", overrides=overrides)

        assert device.policies.target_clock_mhz == 500.0
        assert device.policies.max_clock_mhz == 600.0

    # Invalid overrides cause validation to fail test
    def test_load_device_invalid_override_fails(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))

        overrides = {
            "resources": {
                "luts": -100,  # Negative LUTs
            }
        }

        with pytest.raises(ValueError, match="luts.*positive"):
            registry.load_device("test_device_1", overrides=overrides)

    # Load device nonexistent preset test
    def test_load_device_nonexistent_preset(self, temp_config_dir):
        registry = DeviceRegistry(str(temp_config_dir))

        with pytest.raises(KeyError, match="not found"):
            registry.load_device("nonexistent_preset")

    # Invalid config directory test
    def test_registry_invalid_config_dir(self):
        with pytest.raises(RuntimeError, match="does not exist"):
            DeviceRegistry("/nonexistent/path/to/config")

