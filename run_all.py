"""Generate manuscript Figures 1-9 and supplementary Figures S1-S2."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from crustal_heating.cli import main
    raise SystemExit(main(["all"]))
