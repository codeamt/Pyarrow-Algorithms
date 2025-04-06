import pyarrow as pa

class GeoHasher:
    """Geohash encoder/decoder with PyArrow optimizations.

    This class provides functionality to encode geographic coordinates 
    (latitude, longitude) into geohashes and decode geohashes back into 
    coordinates. It leverages PyArrow for efficient data handling and 
    vectorized operations.

    Geohashes are short alphanumeric strings that represent rectangular 
    areas on the Earth's surface. They are commonly used for spatial 
    indexing, proximity searches, and data visualization.

    This implementation supports variable precision (hash length) 
    and provides methods to calculate neighboring geohashes.

    Attributes:
        precision (int): The desired length of the geohash (1-12).
        BASE32 (pyarrow.Array): A PyArrow array containing the base32 characters.
        BASE32_MAP (dict): A dictionary mapping base32 characters to their indices.

    Example:
        >>> geohasher = GeoHasher(precision=10)
        >>> geohash = geohasher.encode(37.7749, -122.4194)  # Encode coordinates
        >>> coordinates = geohasher.decode(geohash)  # Decode geohash
        >>> neighbors = geohasher.neighbors(geohash)  # Get neighboring geohashes

    """
    BASE32 = pa.array(list('0123456789bcdefghjkmnpqrstuvwxyz'))
    BASE32_MAP = {c:i for i,c in enumerate(BASE32.to_pylist())}
    def __init__(self, precision: int = 10):
        self.precision = min(max(precision, 1), 12)
        self.bits = self.precision * 5
        self.mask = pa.bit_mask(self.bits)
        
    def encode(self, lat: float, lon: float) -> pa.StringScalar:
        """Encode coordinates to geohash"""
        lat = pa.scalar(max(-90.0, min(90.0, lat)))
        lon = pa.scalar(((lon + 180) % 360) - 180)
        bits = pa.BitArrayBuilder()
        lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]

        for i in pa.compute.range(self.bits):
            if i % 2: # Latitude bits
                mid = (lat_range[0] + lat_range[1]) / 2
                bit = lat >= mid
                lat_range[bit] = mid
            else: # Longitude bits
                mid = (lon_range[0] + lon_range[1]) / 2
                bit = lon >= mid
                lon_range[bit] = mid
                bits.append(bit)
                
        return self._pack_bits(bits.finish())

    def decode(self, geohash: pa.StringScalar) -> pa.StructScalar:
        """Decode geohash to coordinates with error margins
        Returns:
        Struct with fields: lon, lat, lon_err, lat_err
        """
        bits = self._unpack_bits(geohash)
        lon_range = pa.array([-180.0, 180.0])
        lat_range = pa.array([-90.0, 90.0])
        for i, bit in enumerate(bits):
            arr, idx = (lon_range, 0) if i%2==0 else (lat_range, 1)
            mid = (arr[0] + arr[1]) / 2
            arr = pa.array([arr[0], mid]) if not bit else pa.array([mid, arr[1]])
            if i%2 == 0: 
                lon_range = arr 
            else: 
                lat_range = arr
        
        return pa.struct([
            ('lon', (lon_range[0] + lon_range[1]) / 2),
            ('lat', (lat_range[0] + lat_range[1]) / 2),
            ('lon_err', (lon_range[1] - lon_range[0]) / 2),
            ('lat_err', (lat_range[1] - lat_range[0]) / 2)
        ])
    
    def _pack_bits(self, bits: pa.BitArray) -> pa.StringScalar:
        """Pack bits into base32 string"""
        chunks = bits.buffers()[1].cast(pa.uint32())
        return pa.compute.utf8_lower(pa.compute.base32_encode(chunks))[:self.precision]

    def _unpack_bits(self, geohash: pa.StringScalar) -> pa.BitArray:
        """Unpack base32 to bit array"""
        decoded = pa.compute.base32_decode(geohash.utf8_upper())
        return pa.BitArray.from_buffers(
            pa.binary(self.bits//8 + 1),
            [None, decoded.buffers()[1].copy()]
        ).mask(self.mask)

    def neighbors(self, geohash: pa.StringScalar) -> pa.StructScalar:
        """
        Calculate all 8 adjacent geohashes with error boundaries
        Args:
        geohash: Input geohash string scalar
        Returns:
        Arrow Struct containing:
        - center: Original geohash coordinates
        - n/nw/ne/e/se/s/sw/w: Neighboring geohashes
        - bounds: Error boundaries for neighbors
        """
        decoded = self.decode(geohash)
        lat, lon = decoded['lat'], decoded['lon']
        lat_err, lon_err = decoded['lat_err'], decoded['lon_err']
        # Calculate step sizes using vectorized operations
        steps = pa.array([
            (lat_err, 0), # north
            (lat_err, lon_err), # ne
            (0, lon_err), # east
            (-lat_err, lon_err), # se
            (-lat_err, 0), # south
            (-lat_err, -lon_err), # sw
            (0, -lon_err), # west
            (lat_err, -lon_err) # nw
        ], type=pa.struct([
            ('dlat', pa.float64()),
            ('dlon', pa.float64())
        ]))

        # Vectorized coordinate calculations
        new_lats = pa.compute.add(lat, steps['dlat'])
        new_lons = pa.compute.add(lon, steps['dlon'])
        # Clamp latitudes and wrap longitudes
        new_lats = pa.compute.clip(new_lats, -90.0, 90.0)
        new_lons = pa.compute.subtract(
            pa.compute.modulo(
                pa.compute.add(new_lons, 180.0),
                360.0
            ),
            180.0
        )

        # Batch encode neighbors
        neighbor_hashes = self.encode(new_lats, new_lons)
        
        return pa.struct([
            ('center', geohash),
            ('n', neighbor_hashes[0]),
            ('ne', neighbor_hashes[1]),
            ('e', neighbor_hashes[2]),
            ('se', neighbor_hashes[3]),
            ('s', neighbor_hashes[4]),
            ('sw', neighbor_hashes[5]),
            ('w', neighbor_hashes[6]),
            ('nw', neighbor_hashes[7]),
            ('bounds', pa.struct([
                ('lat_step', lat_err * 2),
                ('lon_step', lon_err * 2)
            ]))
        ])