try:
    from enum import StrEnum
except ImportError:
    from enum import Enum
    class StrEnum(str, Enum):
        pass

class WatermarkType(StrEnum):
    AAR = "aar"
    KGW = "kgw"
    KTH = "kth"
    PRO = "pro"
