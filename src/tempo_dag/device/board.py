# For describing FPGA Configuration and Hardware
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Resources:
    # Number of Look-Up Tables (LUTS) available
    luts: int

    # Number of Flip-Flops available
    ffs: int

    # Number of DSP blocks available
    dsps: int

    # Number of 36Kb BRAM blocks available
    bram_36k: int

    # Number of 18Kb BRAM blocks available
    bram_18k: int = 0


@dataclass
class Memory:
    # On-chip memory in kilobytes
    on_chip_kb: int

    # External memory bandwidth in Gbps (DDR4, HBM, etc.)
    external_bandwidth_gbps: float

    # External memory latency in nanoseconds
    external_latency_ns: float = 100.0


@dataclass
class IO:
    # PCIe lanes available
    pcie_lanes: int = 16

    # PCIe generation (3, 4, 5)
    pcie_gen: int = 4

    # Other interfaces (e.g., Ethernet, custom)
    other_interfaces: dict[str, str] = field(default_factory=dict)


@dataclass
class Capabilities:
    # Supports 32-bit floating point
    supports_fp32: bool = True

    # Supports 64-bit floating point
    supports_fp64: bool = False

    # Supports bfloat16
    supports_bfloat16: bool = False

    # Supports 8-bit integer
    supports_int8: bool = True

    # Supports 16-bit integer
    supports_int16: bool = True

    # Supports DSP block cascading
    supports_dsp_cascading: bool = True


@dataclass
class Policies:
    # Maximum clock frequency in MHz
    max_clock_mhz: float = 250.0

    # Target clock frequency for designs (may be lower than max)
    target_clock_mhz: float = 200.0

    # Default quantization precision (int8, int16, fp32, etc.)
    default_precision: str = "int8"

    # Power budget in watts (if available)
    power_budget_w: float | None = None


@dataclass
class FPGADevice:
    # Device name (e.g., 'U250')
    name: str

    # FPGA vendor (e.g., 'Xilinx', 'Intel')
    vendor: str

    # FPGA part number
    part_number: str

    # Fabric resources
    resources: Resources

    # Memory configuration
    memory: Memory

    # I/O configuration
    io: IO = field(default_factory=IO)

    # Hardware capabilities
    capabilities: Capabilities = field(default_factory=Capabilities)

    # Design policies
    policies: Policies = field(default_factory=Policies)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FPGADevice":

        resources_data = data.get("resources", {})
        memory_data = data.get("memory", {})
        io_data = data.get("io", {})
        capabilities_data = data.get("capabilities", {})
        policies_data = data.get("policies", {})

        return cls(
            name=data["name"],
            vendor=data["vendor"],
            part_number=data["part_number"],
            resources=Resources(**resources_data),
            memory=Memory(**memory_data),
            io=IO(**io_data),
            capabilities=Capabilities(**capabilities_data),
            policies=Policies(**policies_data),
        )

    def merge_overrides(self, overrides: dict[str, Any]) -> "FPGADevice":

        # Deep copy to avoid mutating original
        device_dict = deepcopy(self.to_dict())

        # Recursively merge overrides
        self._deep_merge(device_dict, overrides)

        return FPGADevice.from_dict(device_dict)

    @staticmethod
    def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> None:

        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                FPGADevice._deep_merge(base[key], value)
            else:
                # Replace scalar, list, or create new key
                base[key] = value

    def validate(self) -> None:

        if not self.name or not isinstance(self.name, str):
            raise ValueError("Device 'name' must be a non-empty string")

        if not self.vendor or not isinstance(self.vendor, str):
            raise ValueError("Device 'vendor' must be a non-empty string")

        if not self.part_number or not isinstance(self.part_number, str):
            raise ValueError("Device 'part_number' must be a non-empty string")

        if self.resources.luts <= 0:
            raise ValueError("Resources 'luts' must be positive")

        if self.resources.ffs <= 0:
            raise ValueError("Resources 'ffs' must be positive")

        if self.resources.dsps < 0:
            raise ValueError("Resources 'dsps' must be non-negative")

        if self.resources.bram_36k < 0:
            raise ValueError("Resources 'bram_36k' must be non-negative")

        if self.resources.bram_18k < 0:
            raise ValueError("Resources 'bram_18k' must be non-negative")

        if self.memory.on_chip_kb <= 0:
            raise ValueError("Memory 'on_chip_kb' must be positive")

        if self.memory.external_bandwidth_gbps < 0:
            raise ValueError("Memory 'external_bandwidth_gbps' must be non-negative")

        if self.memory.external_latency_ns < 0:
            raise ValueError("Memory 'external_latency_ns' must be non-negative")

        if self.io.pcie_lanes < 0:
            raise ValueError("IO 'pcie_lanes' must be non-negative")

        if self.io.pcie_gen not in (3, 4, 5):
            raise ValueError("IO 'pcie_gen' must be 3, 4, or 5")

        if self.policies.max_clock_mhz <= 0:
            raise ValueError("Policies 'max_clock_mhz' must be positive")

        if self.policies.target_clock_mhz <= 0:
            raise ValueError("Policies 'target_clock_mhz' must be positive")

        if self.policies.target_clock_mhz > self.policies.max_clock_mhz:
            raise ValueError(
                "Policies 'target_clock_mhz' cannot exceed 'max_clock_mhz'"
            )

        if (
            self.policies.power_budget_w is not None
            and self.policies.power_budget_w <= 0
        ):
            raise ValueError("Policies 'power_budget_w' must be positive if set")
