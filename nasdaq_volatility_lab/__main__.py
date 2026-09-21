"""Allow python -m nasdaq_volatility_lab alongside the installed CLI."""

from .cli import main

raise SystemExit(main())
