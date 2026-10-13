"""Run the evaluate stage; equivalent to wapa evaluate."""

import sys
from wapa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "evaluate")
    main()
