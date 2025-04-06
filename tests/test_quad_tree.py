import pytest
from hypothesis import given, strategies as st

from data_structures.quad_tree import QuadTree


class TestQuadTree:
    @given(
        points=st.lists(
            st.tuples(
                st.floats(-100, 100), 
                st.floats(-100, 100)
            )
        )
    )
    def test_insert_query(self, points):
        qt = QuadTree((0, 0, 200, 200))
        
        # Insert all points
        for p in points:
            assert qt.insert(p)
        
        # Query entire area
        results = qt.query_region((0, 0, 200, 200))
        assert len(results) == len(points)

    def test_high_density(self):
        qt = QuadTree((0, 0, 1, 1), capacity=4)
        
        # Insert 100 points in same cell
        for _ in range(100):
            qt.insert((0.5, 0.5))
            
        # Should subdivide to max depth
        assert qt.depth >= QuadTree.MAX_DEPTH - 2

    @given(
        st.tuples(
            st.floats(0, 1), 
            st.floats(0, 1)
        ).filter(lambda p: p[0] != p[1])  # Exclude edge
    )
    def test_edge_points(self, point):
        qt = QuadTree((0, 0, 2, 2))
        qt.insert(point)
        
        # Query adjacent regions
        regions = [
            (point[0]-0.1, point[1]-0.1, 0.2, 0.2),
            (point[0]+0.05, point[1]+0.05, 0.2, 0.2)
        ]
        assert sum(len(qt.query_region(r)) for r in regions) >= 1

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=quad-tree",
#        "--cov-report=html:coverage"
#    ])
# --------------------------