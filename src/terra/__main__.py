"""Allow ``python -m terra`` to use the installed command surface."""

from terra.commands import main

if __name__ == "__main__":
    raise SystemExit(main())
