import pyarrow as pa
import pyarrow.compute as pc
from datetime import datetime

class TextOperation:
    """PyArrow-optimized text operation container"""
    def __init__(self):
        self.ops = pa.array([], type=pa.struct([
            ('type', pa.string()),
            ('pos', pa.int64()),
            ('text', pa.string()),
            ('length', pa.int64())
        ]))
        
    def retain(self, n: int):
        """Preserve n characters"""
        self.ops = self.ops.append(pa.struct({
            'type': 'retain',
            'pos': None,
            'text': None,
            'length': n
        }))
        return self
    
    def insert(self, text: str):
        """Insert text at current position"""
        self.ops = self.ops.append(pa.struct({
            'type': 'insert',
            'pos': None,
            'text': text,
            'length': len(text)
        }))
        return self

    def delete(self, text: str):
        """Delete specified text"""
        self.ops = self.ops.append(pa.struct({
            'type': 'delete',
            'pos': None,
            'text': text,
            'length': -len(text)
        }))
        return self

    def compose(self, other: 'TextOperation') -> 'TextOperation':
        """Vectorized operation composition using Arrow compute"""
        combined_ops = pc.concat_arrays([self.ops, other.ops])
        return self._optimize_operations(combined_ops)

    def transform(self, other: 'TextOperation') -> 'TextOperation':
        """Vectorized operation transformation"""
        transformed = []
        our_idx = 0
        their_idx = 0
        curr_our = None
        curr_their = None

        def next_op(ops, idx):
            if idx < len(ops):
                return ops[idx].as_py(), idx + 1
            return None, idx

        while our_idx < len(self.ops) or their_idx < len(other.ops) or curr_our or curr_their:
            if curr_our is None:
                curr_our, our_idx = next_op(self.ops, our_idx)
            if curr_their is None:
                curr_their, their_idx = next_op(other.ops, their_idx)

            if curr_our is None:
                if curr_their:
                    transformed.append(pa.struct(curr_her))
                    curr_their = None
                continue
            if curr_their is None:
                transformed.append(pa.struct(curr_our))
                curr_our = None
                continue

            our_type = curr_our['type']
            their_type = curr_their['type']
            our_len = curr_our['length']
            their_len = curr_their['length']

            # Case 1: Both are retains
            if our_type == 'retain' and their_type == 'retain':
                min_len = min(our_len, their_len)
                transformed.append(pa.struct([
                    ('type', 'retain'),
                    ('pos', None),
                    ('text', None),
                    ('length', min_len)
                ]))
                if our_len > min_len:
                    curr_our['length'] = our_len - min_len
                else:
                    curr_our = None
                if their_len > min_len:
                    curr_their['length'] = their_len - min_len
                else:
                    curr_their = None

            # Case 2: Insert vs Retain
            elif our_type == 'insert' and their_type == 'retain':
                transformed.append(pa.struct([
                    ('type', 'insert'),
                    ('pos', None),
                    ('text', curr_our['text']),
                    ('length', curr_our['length'])
                ]))
                curr_our = None
                if their_len > our_len:
                    curr_their['length'] -= our_len
                else:
                    curr_their = None

            # Case 3: Retain vs Insert
            elif our_type == 'retain' and their_type == 'insert':
                transformed.append(pa.struct([
                    ('type', 'insert'),
                    ('pos', None),
                    ('text', curr_their['text']),
                    ('length', curr_their['length'])
                ]))
                curr_their = None
                curr_our['length'] += curr_their['length']

            # Case 4: Delete vs Retain
            elif our_type == 'delete' and their_type == 'retain':
                transformed.append(pa.struct([
                    ('type', 'delete'),
                    ('pos', None),
                    ('text', curr_our['text']),
                    ('length', curr_our['length'])
                ]))
                curr_our = None
                if their_len > -curr_our['length']:
                    curr_their['length'] += curr_our['length']
                else:
                    curr_their = None

            # Case 5: Retain vs Delete
            elif our_type == 'retain' and their_type == 'delete':
                transformed.append(pa.struct([
                    ('type', 'delete'),
                    ('pos', None),
                    ('text', curr_their['text']),
                    ('length', curr_their['length'])
                ]))
                curr_their = None
                if our_len > -curr_their['length']:
                    curr_our['length'] += curr_their['length']
                else:
                    curr_our = None

            # Handle other cases (insert/delete, delete/delete)
            else:
                transformed.append(pa.struct(curr_our))
                transformed.append(pa.struct(curr_their))
                curr_our = None
                curr_her = None

        # Create new TextOperation with transformed ops
        transformed_ops = pa.array(transformed, type=self.ops.type)
        return TextOperation().from_array(transformed_ops)

    def from_array(self, arr: pa.Array) -> 'TextOperation':
        self.ops = arr
        return self

    def _optimize_operations(self, ops: pa.Array) -> pa.Array:
        """Merge adjacent operations of the same type"""
        return pc.list_flatten(
            pc.aggregate(
                pc.run_length_encode(pc.field('type')),
                pc.field('ops')
            ).cast(ops.type)
        )

class OTVersionControl:
    """PyArrow-optimized version control system for collaborative editing"""
    def __init__(self):
        self.history = pa.Table.from_arrays(
            arrays=[[], [], [], []],
            names=['timestamp', 'client_id', 'version', 'operation'],
            schema=pa.schema([
                ('timestamp', pa.timestamp('ns')),
                ('client_id', pa.string()),
                ('version', pa.int64()),
                ('operation', TextOperation().ops.type)
            ])
        )
        
    def apply_operation(self, client_id: str, operation: TextOperation):
        """Apply and store operation with version control"""
        # Transform against concurrent operations
        server_ops = self.history.filter(
            pc.field('version') >= operation.metadata['base_version']
        ).column('operation')
        
        transformed_op = operation
        for sop in server_ops:
            transformed_op = transformed_op.transform(sop)
            
        # Update current text
        self.current_text = self._apply_vectorized(
            self.current_text,
            transformed_op.ops
        ) 

        # Add to history
        self.history = self.history.append(
            pa.record_batch([
                pa.array([datetime.now().isoformat()], pa.timestamp('ns')),
                pa.array([client_id], pa.string()),
                pa.array([self.version], pa.int64()),
                pa.array([transformed_op.ops], transformed_op.ops.type)
            ])
        )
        self.version += 1

    def _apply_vectorized(self, text: pa.Array, ops: pa.Array) -> pa.Array:
        """Apply operations using Arrow vectorized string functions"""
        result = text
        pos = 0
        for op in ops:
            op_type = op['type'].as_py()
            length = op['length'].as_py()
            content = op['text'].as_py()
            
            if op_type == 'retain':
                pos += length
            elif op_type == 'insert':
                result = pc.utf8_insert(result, pos, content)
                pos += length
            elif op_type == 'delete':
                result = pc.utf8_slice(result, 0, pos) + pc.utf8_slice(result, pos + length)
        return result

    def get_state(self) -> pa.StringArray:
        """Get current text state as Arrow StringArray"""
        return self.current_text

    def get_version_history(self) -> pa.Table:
        """Get complete operation history as Arrow Table"""
        return self.history.sort_by('version')

    def compress_history(self):
        """Optimize history storage by composing operations"""
        compressed_ops = pc.aggregate(
            self.history.column('operation'),
            'list',
            options=pc.AggregateOptions(
                skip_nulls=True,
                mode='combine'
            )
        )
        self.history = pa.Table.from_arrays(
            [
                pa.array([datetime.now().isoformat()], pa.timestamp('ns')),
                pa.array(['system'], pa.string()),
                pa.array([self.version], pa.int64()),
                compressed_ops
            ],
            names=self.history.schema.names
        )