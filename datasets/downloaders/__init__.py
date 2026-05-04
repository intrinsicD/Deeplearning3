"""Auto-import all downloaders so they self-register."""

from . import ucf101      # noqa: F401
from . import hmdb51      # noqa: F401
from . import vggsound    # noqa: F401
from . import audioset    # noqa: F401
from . import kinetics    # noqa: F401
from . import miradata    # noqa: F401
from . import avspeech    # noqa: F401

