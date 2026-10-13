"""Run the acquire stage; equivalent to wapa acquire."""

import sys
from wapa.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "acquire")
    main()
