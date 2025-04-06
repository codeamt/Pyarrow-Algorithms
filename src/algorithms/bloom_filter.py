import hashlib
import math 
import pyarrow as pa

class BloomFilter:
    """Space-efficient probabilistic membership tester using PyArrow buffers
        Args:
        capacity: Expected maximum number of elements
        error_rate: Acceptable false positive rate 01
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