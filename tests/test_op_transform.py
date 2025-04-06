import pytest
from hypothesis import given, strategies as st

from src.algorithms.op_transform import TextOperation, OTVersionControl

class TestOperationalTransformation:
    @given(
        ops=st.lists(
            st.one_of(
                st.tuples(st.just("ins"), st.text(), st.integers(0, 100)),
                st.tuples(st.just("del"), st.integers(1, 10), st.integers(0, 100))
            )
        )
    )
    def test_convergence(self, ops):
        doc1 = OTVersionControl()
        doc2 = OTVersionControl()
        
        # Apply same operations in different orders
        for op in ops:
            doc1.apply(op)
            transformed = doc2.transform(op)
            doc2.apply(transformed)
            
        assert doc1.get_state() == doc2.get_state()

    def test_conflict_resolution(self):
        doc = OTVersionControl()
        
        # Concurrent inserts at position 0
        op1 = TextOperation().insert("A")
        op2 = TextOperation().insert("B")
        
        doc.apply(op1)
        doc.apply(doc.transform(op2))
        
        assert doc.get_state().to_pylist() == ["B", "A"]  # Or ["A", "B"]

    @given(st.text(), st.text())
    def test_transform_properties(self, a, b):
        opA = TextOperation().insert(a)
        opB = TextOperation().insert(b)
        
        # Should commute: opA + opB' = opB + opA'
        transformedB = opA.transform(opB)
        transformedA = opB.transform(opA)
        
        doc1 = opA.compose(transformedB)
        doc2 = opB.compose(transformedA)
        
        assert doc1.apply("") == doc2.apply("")

# --------------------------
# Running Tests
# --------------------------
#if __name__ == "__main__":
#    pytest.main([
#        "-v", 
#        "--hypothesis-show-statistics",
#        "--cov=op-transform",
#        "--cov-report=html:coverage"
#    ])
# --------------------------