"""Make singlish_labelling importable whether it lives at repo root or in src/."""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_root, os.path.join(_root, "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
