__all__ = ['openbt']


__version__ = "0.0.6"

import sys

if sys.version_info[0] == 3 and sys.version_info[1] < 6:
    raise ImportError("Python Version 3.6 or above is required for fdasrsf.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys
