import pyarrow as pa
import pyarrow.compute as pc

class ArrowRayCaster:
    """PyArrow-optimized ray casting with bounding box optimization
    Features:
    - Vectorized batch processing of points
    - SIMD-accelerated bounding box checks
    - Columnar memory layout for GPU compatibility
    - Precomputed polygon metadata
    """
    def __init__(self, polygons: pa.StructArray):
        """
        Initialize with polygons structured as:
        [
            {
                'bbox': {min_x, max_x, min_y, max_y},
                'edges': [
                    {x1, y1, x2, y2,
                    ...
                ]
            },
            ...
        ]
        """
        self.polygons = self._preprocess_polygons(polygons)
        
    def _preprocess_polygons(self, raw_polygons: pa.StructArray) -> pa.StructArray:
        """Precompute bounding boxes and edge parameters"""
        def process_poly(poly):
            vertices = poly['vertices']
            x = vertices['x']
            y = vertices['y']
            edges = pc.list_flatten(pc.make_struct(
                pc.list_slice(x, 0, -1), # x1
                pc.list_slice(y, 0, -1), # y1
                pc.list_slice(x, 1, None), # x2
                pc.list_slice(y, 1, None) # y2
            ).combine_chunks())
        
            return pa.struct({
                'bbox': pa.struct({
                    'min_x': pc.min(x),
                    'max_x': pc.max(x),
                    'min_y': pc.min(y),
                    'max_y': pc.max(y)
                }),
                'edges': edges
            })
        
        return pc.vector_unary_function(process_poly, raw_polygons)

    def contains(self, points: pa.StructArray) -> pa.BooleanArray:
        """Batch check point-in-polygon for multiple points"""
        results = pc.repeat(False, len(points))
        for poly in self.polygons:
            # Bounding box quick rejection
            in_bbox = pc.and_(
                pc.and_(
                    pc.greater_equal(points['x'], poly['bbox']['min_x']),
                    pc.less_equal(points['x'], poly['bbox']['max_x'])
                ),
                pc.and_(
                    pc.greater_equal(points['y'], poly['bbox']['min_y']),
                    pc.less_equal(points['y'], poly['bbox']['max_y'])
                )
            )
            
            # Vectorized edge intersection checks
            crossings = pc.sum(
                self._check_edges(points, poly['edges']),
                skip_nulls=True
            )
        
            # Update results using bitwise XOR for odd/even check
            results = pc.if_else(
                in_bbox,
                pc.bit_wise_xor(results, pc.mod(crossings, 2)),
                results
            )
            return results

    def _check_edges(self, points: pa.StructArray, edges: pa.StructArray) -> pa.BooleanArray:
        """Vectorized edge intersection check"""
        # Broadcast points against edges
        points_expanded = pc.repeat(points, len(edges))
        edges_repeated = pc.repeat(edges, len(points), axis=1)
        # Unpack coordinates
        p_x = points_expanded['x']
        p_y = points_expanded['y']
        e = edges_repeated
        # Edge parameters
        y1 = e['y1']
        y2 = e['y2']
        x1 = e['x1']
        x2 = e['x2']
        
        # Calculate intersection parameters
        with pc.ExpressionContext(preserve_floating_point=True):
            t_numer = (y2 - y1) * (x1 - p_x)
            t_denom = (y2 - y1) * (x1 - x2)
            u_numer = (p_y - y1) * (x1 - x2)
            denominator = (y2 - y1) * (x1 - x2) - (x2 - x1) * (y1 - y2)
            
            # Valid intersection conditions
            valid = pc.and_(
                pc.and_(
                    pc.greater(p_y, pc.min(y1, y2)),
                    pc.less_equal(p_y, pc.max(y1, y2))
                ),
                pc.and_(
                    pc.greater(p_x, pc.min(x1, x2)),
                    pc.less(p_x, pc.max(x1, x2))

                )
            )
            
            # Edge crossing conditions
            crosses = pc.and_(
                pc.greater(t_numer / t_denom, 0),
                pc.less(u_numer / denominator, 1)
            )
            return pc.and_(valid, crosses)