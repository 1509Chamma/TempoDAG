# Loads and manages device presets from JSON

import json
from pathlib import Path
from typing import Any

from .board import FPGADevice


class DeviceRegistry:
    def __init__(self, config_dir: str | None = None):

        if config_dir is None:
            package_root = Path(__file__).resolve().parents[3]
            config_dir = str(package_root / "configs" / "devices")

        self.config_dir = Path(config_dir)
        self._presets: dict[str, dict[str, Any]] = {}
        self._load_presets()

    def _load_presets(self) -> None:

        if not self.config_dir.exists():
            raise RuntimeError(
                f"Config directory does not exist: {self.config_dir}\n"
                "Please create it or pass a valid config_dir to DeviceRegistry."
            )

        for file in self.config_dir.glob("*.json"):
            try:
                with open(file) as f:
                    preset = json.load(f)
                    if preset and "name" in preset:
                        preset_name = preset["name"]
                        self._presets[preset_name] = preset
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"Failed to load preset {file}: {e}") from e

    def list_presets(self) -> list:
        return sorted(self._presets.keys())

    def get_preset(self, preset_name: str) -> dict[str, Any]:

        if preset_name not in self._presets:
            available = ", ".join(self.list_presets()) or "(none)"
            raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")
        return dict(self._presets[preset_name])

    def load_device(
        self,
        preset_name: str,
        overrides: dict[str, Any] | None = None,
    ) -> FPGADevice:

        preset_config = self.get_preset(preset_name)

        device = FPGADevice.from_dict(preset_config)

        if overrides:
            device = device.merge_overrides(overrides)

        device.validate()

        return device
