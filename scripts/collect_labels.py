"""Run the collect stage; equivalent to wapa collect."""

import sys
from wapa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "collect")
    main()
