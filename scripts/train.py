"""Run the train stage; equivalent to wapa train."""

import sys
from wapa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "train")
    main()
