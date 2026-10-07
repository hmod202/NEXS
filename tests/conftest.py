import os
import shutil
import tempfile
from pathlib import Path

# Isolate config + DB before any nexs module is imported.
_tmp = Path(tempfile.mkdtemp())
shutil.copytree(Path(__file__).resolve().parent.parent / "config", _tmp / "config")
os.environ["NEXS_CONFIG_DIR"] = str(_tmp / "config")
os.environ["NEXS_DB"] = str(_tmp / "nexs.db")
os.environ["NEXS_BROKER"] = "sim"
os.environ.pop("NEXS_TOKEN", None)
