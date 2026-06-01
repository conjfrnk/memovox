"""Importable from any test module to guarantee ``src/`` is on ``sys.path``.

Allows running either ``python -m unittest discover -s tests`` (which imports the
``tests`` package and runs ``__init__``) or a single test file directly.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
