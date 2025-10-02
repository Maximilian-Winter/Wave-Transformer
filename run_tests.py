#!/usr/bin/env python
"""
Convenience script to run tests with common configurations.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Skip slow tests
    python run_tests.py --cov        # Run with coverage
    python run_tests.py --verbose    # Verbose output
"""

import sys
import subprocess

def main():
    args = sys.argv[1:]

    cmd = ["python", "-m", "pytest", "tests/"]

    if "--fast" in args:
        cmd.extend(["-m", "not slow"])
        args.remove("--fast")

    if "--cov" in args:
        cmd.extend(["--cov=src/wave_transformer", "--cov-report=html", "--cov-report=term"])
        args.remove("--cov")

    if "--verbose" in args:
        cmd.append("-vv")
        args.remove("--verbose")
    else:
        cmd.append("-v")

    # Add any remaining arguments
    cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
