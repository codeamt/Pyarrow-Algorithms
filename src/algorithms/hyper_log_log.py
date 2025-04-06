import math
import hashlib
import pyarrow as pa

class HyperLogLog:
    """Cardinality estimator using PyArrow arrays
    Args:
    precision: Accuracy vs memory tradeoff (4-16)
    """
    def __init__(self, precision: int = 12):
        self.p = precision
        self.m = 1 << precision
        self.reg = pa.array([0]*self.m, type=pa.uint8())
        
    def add(self, item: str):
        """Add item to cardinality estimate"""
        h = int.from_bytes(hashlib.sha256(item.encode()).digest(), 'big')
        idx = h & (self.m-1)
        self.reg[idx] = max(self.reg[idx], 64 - (h >> self.p).bit_length())

    def count(self) -> int:
        """Get cardinality estimate"""
        # epsilon
        e = 0.7213/(1+1.079/self.m) * self.m**2 / sum(2**-v for v in self.reg)
        return int(e if e > 2.5*self.m else self.m * math.log(self.m/self.reg.null_count))