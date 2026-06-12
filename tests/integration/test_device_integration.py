from tempo_dag.device import FPGADevice, Memory, Policies, Resources


class TestFPGADeviceIntegration:
    # Basic validation in integration context
    def test_validate_basic_device(self):
        basic_device = FPGADevice(
            name="test_device",
            vendor="TestVendor",
            part_number="TEST-001",
            resources=Resources(luts=100000, ffs=200000, dsps=500, bram_36k=100),
            memory=Memory(on_chip_kb=4096, external_bandwidth_gbps=19.2),
        )

        basic_device.validate()

    # Full workflow test: create, serialize, deserialize, override, validate
    def test_full_workflow(self):
        device1 = FPGADevice(
            name="u250_base",
            vendor="Xilinx",
            part_number="xcu250-figd2104-2L",
            resources=Resources(luts=1728000, ffs=3456000, dsps=6144, bram_36k=2688),
            memory=Memory(
                on_chip_kb=86016, external_bandwidth_gbps=76.8, external_latency_ns=10.0
            ),
            policies=Policies(
                max_clock_mhz=300.0, target_clock_mhz=250.0, power_budget_w=75.0
            ),
        )

        device_dict = device1.to_dict()
        device2 = FPGADevice.from_dict(device_dict)

        overrides = {
            "name": "u250_low_power",
            "policies": {
                "target_clock_mhz": 150.0,
                "power_budget_w": 40.0,
            },
        }
        device3 = device2.merge_overrides(overrides)

        device1.validate()
        device2.validate()
        device3.validate()

        assert device3.name == "u250_low_power"
        assert device3.policies.target_clock_mhz == 150.0
        assert device3.policies.power_budget_w == 40.0
        assert device3.resources.dsps == 6144

