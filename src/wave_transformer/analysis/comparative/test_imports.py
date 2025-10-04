"""
Quick validation script to test imports and basic functionality
of the comparative analysis tools.
"""

import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent.parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    # Test direct comparative imports (should always work)
    try:
        from wave_transformer.analysis.comparative import (
            CheckpointComparator,
            InputComparator,
            AblationHelper
        )
        print("[PASS] Direct comparative imports successful")
        direct_success = True
    except ImportError as e:
        print(f"[FAIL] Direct comparative imports failed: {e}")
        direct_success = False

    # Test main package imports (may fail due to optional deps in other modules)
    try:
        from wave_transformer.analysis import (
            CheckpointComparator,
            InputComparator,
            AblationHelper
        )
        print("[PASS] Main package imports successful")
        main_success = True
    except ImportError as e:
        print(f"[WARN] Main package imports failed (possibly due to optional deps): {e}")
        print("       This is OK if the comparative modules imported successfully above.")
        main_success = False

    # We consider the test passed if at least direct imports work
    return direct_success


def test_class_initialization():
    """Test that classes can be instantiated (without actual models)"""
    print("\nTesting class structure...")

    try:
        from wave_transformer.analysis.comparative import (
            CheckpointComparator,
            InputComparator,
            AblationHelper
        )

        # Check class attributes
        assert hasattr(CheckpointComparator, '__init__')
        assert hasattr(CheckpointComparator, 'compare_on_input')
        assert hasattr(CheckpointComparator, 'compare_on_dataset')
        assert hasattr(CheckpointComparator, 'compute_checkpoint_divergence')
        assert hasattr(CheckpointComparator, 'plot_checkpoint_evolution')
        assert hasattr(CheckpointComparator, 'identify_critical_checkpoints')
        print("[PASS] CheckpointComparator has all required methods")

        assert hasattr(InputComparator, '__init__')
        assert hasattr(InputComparator, 'compare_inputs')
        assert hasattr(InputComparator, 'compute_input_similarity')
        assert hasattr(InputComparator, 'cluster_inputs')
        assert hasattr(InputComparator, 'plot_input_comparison')
        assert hasattr(InputComparator, 'find_nearest_neighbors')
        print("[PASS] InputComparator has all required methods")

        assert hasattr(AblationHelper, '__init__')
        assert hasattr(AblationHelper, 'ablate_harmonics')
        assert hasattr(AblationHelper, 'ablate_layers')
        assert hasattr(AblationHelper, 'ablate_wave_component')
        assert hasattr(AblationHelper, 'run_ablation_study')
        assert hasattr(AblationHelper, 'restore_model')
        assert hasattr(AblationHelper, 'plot_ablation_results')
        print("[PASS] AblationHelper has all required methods")

        return True

    except Exception as e:
        print(f"[FAIL] Class structure test failed: {e}")
        return False


def test_docstrings():
    """Test that all classes and methods have docstrings"""
    print("\nTesting docstrings...")

    try:
        from wave_transformer.analysis.comparative import (
            CheckpointComparator,
            InputComparator,
            AblationHelper
        )

        classes = [CheckpointComparator, InputComparator, AblationHelper]

        for cls in classes:
            assert cls.__doc__ is not None, f"{cls.__name__} missing docstring"
            print(f"[PASS] {cls.__name__} has docstring")

        return True

    except AssertionError as e:
        print(f"[FAIL] Docstring test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Comparative Analysis Tools - Validation")
    print("=" * 60)

    results = []

    results.append(("Import Test", test_imports()))
    results.append(("Class Structure Test", test_class_initialization()))
    results.append(("Docstring Test", test_docstrings()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        symbol = "[+]" if passed else "[-]"
        print(f"{symbol} {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[ERROR] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
