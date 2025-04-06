import pyarrow as pa
import pyarrow.compute as pc

class QuadTree:
    """PyArrow-optimized spatial index with region query capabilities
    Args:
    boundary: (x, y, width, height) as Arrow StructScalar
    capacity: Max points per node before subdivision
    depth: Current tree depth (internal use)
    """
    def __init__(self, boundary: pa.StructScalar, capacity: int = 4, depth: int = 0):
        self.boundary = boundary
        self.capacity = capacity
        self.depth = depth
        self.points = pa.array([], type=pa.struct([
            ('x', pa.float64()),
            ('y', pa.float64()),
            ('data', pa.string())
        ]))
        self.children = pa.array([], type=pa.struct([
            ('nw', pa.struct([('x', pa.float64()), ('y', pa.float64()),
                              ('width', pa.float64()), ('height', pa.float64())])),
            ('ne', pa.struct([('x', pa.float64()), ('y', pa.float64()),
                              ('width', pa.float64()), ('height', pa.float64())])),
            ('sw', pa.struct([('x', pa.float64()), ('y', pa.float64()),
                              ('width', pa.float64()), ('height', pa.float64())])),
            ('se', pa.struct([('x', pa.float64()), ('y', pa.float64()),
                              ('width', pa.float64()), ('height', pa.float64())]))
        ]))

    def insert(self, point: pa.StructScalar):
        """Insert point into spatial index"""
        if not self._rect_contains(self.boundary, point):
            return False
        
        if len(self.points) < self.capacity and len(self.children) == 0:
            self.points = self.points.append(point)
            return True
        
        if len(self.children) == 0:
            self._subdivide()
        return (self.children['nw'].insert(point) or
                 self.children['ne'].insert(point) or
                 self.children['sw'].insert(point) or
                 self.children['se'].insert(point))

    def _subdivide(self):
        """Create child nodes using Arrow vectorized operations"""
        x, y, w, h = (
            self.boundary['x'].as_py(),
            self.boundary['y'].as_py(),
            self.boundary['width'].as_py(),
            self.boundary['height'].as_py()
        )
        hw, hh = w/2, h/2
        self.children = pa.array([{
            'nw': {'x': x - hw/2, 'y': y + hh/2, 'width': hw, 'height': hh},
            'ne': {'x': x + hw/2, 'y': y + hh/2, 'width': hw, 'height': hh},
            'sw': {'x': x - hw/2, 'y': y - hh/2, 'width': hw, 'height': hh},
            'se': {'x': x + hw/2, 'y': y - hh/2, 'width': hw, 'height': hh}
        }], type=self.children.type)

        for p in self.points:
            (self.children['nw'].insert(p) or
            self.children['ne'].insert(p) or
            self.children['sw'].insert(p) or
            self.children['se'].insert(p))
            self.points = pa.array([], type=self.points.type)

    def query_region_of_interest(self, region: pa.StructScalar) -> pa.Array:
        """Query points within rectangular region using Arrow compute"""
        results = pa.array([], type=self.points.type)
        if not self._rect_intersects(self.boundary, region):
            return results
        
        # Check points in this node
        in_region = pc.and_(
            pc.and_(
                pc.greater_equal(self.points.field('x'), region['x'] - region['width']/2),
                pc.less_equal(self.points.field('x'), region['x'] + region['width']/2)
            ),
            pc.and_(
                pc.greater_equal(self.points.field('y'), region['y'] - region['height']/2),
                pc.less_equal(self.points.field('y'), region['y'] + region['height']/2)
            )
        )
        results = results.append(self.points.filter(in_region))

        # Query children
        if len(self.children) > 0:
            for child in ['nw', 'ne', 'sw', 'se']:
                results = results.append(
                    self.children[child].query_region_of_interest(region)
                )
                return results

    def _rect_contains(self, rect: pa.StructScalar, point: pa.StructScalar) -> pa.BooleanScaler:
        """Arrow-vectorized containment check"""
        return pc.and_(
            pc.and_(
                pc.greater_equal(point['x'], rect['x'] - rect['width']/2),
                pc.less_equal(point['x'], rect['x'] + rect['width']/2)
            ),
            pc.and_(
                pc.greater_equal(point['y'], rect['y'] - rect['height']/2),
                pc.less_equal(point['y'], rect['y'] + rect['height']/2)
            )
        )

    def _rect_intersects(self, a: pa.StructScalar, b: pa.StructScalar) -> pa.BooleanScaler:
        """Arrow-vectorized intersection check"""
        return pc.and_(
            pc.less_equal(
                pc.abs(pc.subtract(a['x'], b['x'])),
                (a['width'] + b['width']) / 2
            ),
            pc.less_equal(
                pc.abs(pc.subtract(a['y'], b['y'])),
                (a['height'] + b['height']) / 2
            )
        )