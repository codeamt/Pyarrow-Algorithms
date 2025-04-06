import pytest
from hypothesis import given, strategies as st

from src.algorithms.geo_hash import GeoHasher

class TestGeohash:
    @given(
        st.floats(-90, 90),
        st.floats(-180, 180),
        st.integers(1, 12)
    )
    def test_geohash_roundtrip(self, lat, lon, precision):
        hasher = GeoHasher()
        gh = hasher.encode(lat, lon, precision)
        decoded = hasher.decode(gh)
        # Check within geohash error bounds
        assert abs(lat - decoded.lat) <= decoded.lat_err
        assert abs(lon - decoded.lon) <= decoded.lon_err

def test_neighbors_consistency(self):
    hasher = GeoHasher()
    neighbors = hasher.neighbors("u4pruy")
    assert len(neighbors) == 8
    assert all(hasher.decode(n).distance < 2*hasher.decode("u4pruy").error for n in neighbors)


# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=geo-hasher",
#        "--cov-report=html:coverage"
#    ])
# --------------------------


