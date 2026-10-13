"""Run the generate stage; equivalent to wapa generate."""

import sys
from wapa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "generate")
    main()
