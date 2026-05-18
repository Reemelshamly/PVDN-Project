#!/usr/bin/env python3
import sys
import json
from pathlib import Path

inpath = Path(sys.argv[1]) if len(sys.argv) > 1 else None

out = {"message": "stub processed", "input": str(inpath)}

print(json.dumps(out))
