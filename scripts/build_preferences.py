"""Run the preferences stage; equivalent to wapa preferences."""

import sys
from wapa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "preferences")
    main()
