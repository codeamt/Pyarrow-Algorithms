import pytest
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from hypothesis import given, strategies as st

from data_structures.bloom_filter import BloomFilter

class TestBloomFilter:
    @given(st.integers(1, 1000), st.floats(0.01, 0.2))
    def test_bloom_filter_basic(self, capacity, error_rate):
        bf = BloomFilter(capacity, error_rate)
        items = [str(i) for i in range(capacity)]
        
        for item in items:
            bf.add(item)
        
        # Verify no false negatives
        for item in items:
            assert item in bf
        
        # Check false positive rate
        false_positives = sum(1 for i in range(capacity) 
                            if str(i + capacity) in bf)
        fp_rate = false_positives / capacity
        assert fp_rate <= error_rate * 1.5  # Allow 50% margin

    def test_bloom_filter_edge_cases(self):
        bf = BloomFilter(10, 0.1)
        assert "missing" not in bf
        with pytest.raises(OverflowError):
            for i in range(1000):
                bf.add(str(i))


# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=bloom-filter",
#        "--cov-report=html:coverage"
#    ])
# --------------------------