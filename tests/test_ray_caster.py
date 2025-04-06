import pytest
import pyarrow as pa
from hypothesis import given, strategies as st

from src.algorithms.ray_caster import ArrowRayCaster

class TestRayCasting:
    def test_polygon_containment(self):
        square = pa.array([(0,0), (10,0), (10,10), (0,10)])
        caster = ArrowRayCaster(square)
        
        points = pa.array([
            (5,5), (15,15), (-1,5), (5,-1)
        ])
        
        results = caster.contains(points)
        assert results.to_pylist() == [True, False, False, False]

    def test_complex_polygons(self):
        star = pa.array([(0,0), (2,5), (5,0), (0,3), (5,3)])
        caster = ArrowRayCaster(star)
        assert caster.contains((1,1)) == True

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=ray-caster",
#        "--cov-report=html:coverage"
#    ])
# --------------------------