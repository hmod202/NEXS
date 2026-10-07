"""Hot-reloadable YAML config: re-read whenever the file changes on disk."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Load .env (KEY=VALUE lines) so the same setup works on Windows without a shell; real env vars win.
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        value = value.split(" #")[0].strip()
        if sep and value and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value)

CONFIG_DIR = Path(os.environ.get("NEXS_CONFIG_DIR", ROOT / "config"))


class YamlFile:
    def __init__(self, name: str):
        self.path = CONFIG_DIR / name
        self._mtime = None
        self._data: dict = {}

    def get(self) -> dict:
        mtime = self.path.stat().st_mtime
        if mtime != self._mtime:
            self._data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            self._mtime = mtime
        return self._data

    def save(self, data: dict) -> None:
        # ponytail: comments in the file are lost on save; use ruamel.yaml if that matters.
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        tmp.replace(self.path)
        self._mtime = None


agents_cfg = YamlFile("agents.yaml")
risk_cfg = YamlFile("risk.yaml")
