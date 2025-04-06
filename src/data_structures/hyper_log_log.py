import math
import hashlib
import pyarrow as pa

class HyperLogLog:
    """Cardinality estimator using PyArrow arrays.

    This class implements the HyperLogLog algorithm, a probabilistic data 
    structure used for estimating the cardinality (number of unique elements) 
    of a set. It leverages PyArrow arrays for efficient storage and 
    computation.

    HyperLogLog provides a space-efficient way to approximate the cardinality 
    of large datasets with high accuracy, using a fixed amount of memory 
    regardless of the actual number of unique elements.

    The accuracy of the estimate is controlled by the `precision` parameter, 
    which determines the number of registers used internally. Higher precision 
    leads to better accuracy but requires more memory.

    Attributes:
        p (int): The precision parameter (4-16).
        m (int): The number of registers, calculated as 2 raised to the power of `p`.
        reg (pyarrow.Array): A PyArrow array storing the register values.

    Example:
        >>> hll = HyperLogLog(precision=12)
        >>> hll.add("element1")
        >>> hll.add("element2")
        >>> hll.add("element1")  # Adding duplicates doesn't affect the count
        >>> estimated_count = hll.count()  # Get the estimated cardinality

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