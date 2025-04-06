import hashlib
import math 
import pyarrow as pa

class BloomFilter:
    """A space-efficient probabilistic data structure for membership testing.

    Bloom filters are used to test whether an element is a member of a set. 
    They are probabilistic in nature, meaning there is a small chance of false positives 
    (indicating an element is present when it's not), but no false negatives 
    (indicating an element is not present when it is).

    This implementation utilizes PyArrow buffers for efficient bit array management.

    Attributes:
        capacity (int): The expected maximum number of elements to be added to the filter.
        error_rate (float): The desired false positive rate (between 0 and 1).
        size (int): The size of the underlying bit array, calculated based on capacity and error rate.
        hash_count (int): The number of hash functions used, calculated based on capacity and size.
        bits (pyarrow.BufferOutputStream): The PyArrow buffer storing the bit array.

    Example:
        >>> bf = BloomFilter(capacity=1000, error_rate=0.01)
        >>> bf.add("hello")
        >>> "hello" in bf
        True
        >>> "world" in bf
        False  # Potentially, with a small probability of being True (false positive)

    """
    def __init__(self, capacity: int, error_rate: float):
        self.size = self._calc_size(capacity, error_rate)
        self.hash_count = self._calc_hash_count(capacity, self.size)
        self.bits = pa.BufferOutputStream()
        self.bits.write(b'\x00' * self.size)
        
    def add(self, item: str):
        """Insert item into filter"""
        h = int.from_bytes(hashlib.blake2s(item.encode()).digest(), 'big')
        for s in range(self.hash_count):
            idx = (h + s*1299721) % self.size
            self.bits.set_bit(idx)
            
    def __contains__(self, item: str) -> bool:
        """Check item membership"""
        h = int.from_bytes(hashlib.blake2s(item.encode()).digest(), 'big')
        return all(self.bits.get_bit((h+s*1299721)%self.size) for s in range(self.hash_count))
        

    @staticmethod
    def _calc_size(n: int, p: float) -> int:
        return math.ceil(-(n * math.log(p)) / (math.log(2)**2))
                                               
    @staticmethod
    def _calc_hash_count(n: int, m: int) -> int:
        return math.ceil((m/n) * math.log(2))