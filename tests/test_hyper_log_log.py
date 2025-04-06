import pytest
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from hypothesis import given, strategies as st

from data_structures.hyper_log_log import HyperLogLog

class TestHyperLogLog:
    @given(st.lists(st.uuids(), min_size=1000, max_size=100000))
    def test_hll_accuracy(self, items):
        hll = HyperLogLog()
        for item in items:
            hll.add(str(item))
        
        estimate = hll.count()
        actual = len(set(items))
        error = abs(estimate - actual) / actual
        
        assert error < 0.02  # 2% error margin

    def test_hll_merge(self):
        hll1 = HyperLogLog()
        hll2 = HyperLogLog()
        
        hll1.add("a"); hll1.add("b")
        hll2.add("b"); hll2.add("c")
        
        merged = hll1.merge(hll2)
        assert abs(merged.count() - 3) <= 1

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=hyper-log-log",
#        "--cov-report=html:coverage"
#    ])
# --------------------------