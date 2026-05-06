import sys

from vespainv.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
