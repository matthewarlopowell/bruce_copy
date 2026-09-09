from setuptools import setup, Extension, find_packages
import sys
import os
import numpy
import platform
import subprocess

# Detect platform and compiler
system = platform.system().lower()
machine = platform.machine().lower()
compiler = os.environ.get("CC", "")
is_clang = "clang" in compiler or system == "darwin"

# Base compile args
compile_args = ["-std=c99" , "-O3", "-ffast-math"]
link_args = []

# Try to enable OpenMP safely
if is_clang:
    # macOS Clang (Xcode) often lacks OpenMP, but Homebrew LLVM supports it
    compile_args += ["-Xpreprocessor", "-fopenmp"]
    link_args += ["-lomp"]

    # Add Homebrew LLVM paths if they exist
    omp_include = "/usr/local/include"
    omp_lib = "/usr/local/lib"
    if os.path.exists(omp_include):
        compile_args.append(f"-I{omp_include}")
    if os.path.exists(omp_lib):
        link_args.append(f"-L{omp_lib}")
else:
    # GCC or compatible
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]

# Add architecture-specific optimizations safely
if not is_clang:
    compile_args.append("-march=native")
else:
    # Only use x86-specific flags on Intel Macs
    if machine in ("x86_64", "amd64"):
        compile_args.append("-mavx2")
    # For ARM (Apple Silicon), prefer NEON or SVE (auto-vectorization handled by compiler)
    elif machine in ("arm64", "aarch64"):
        compile_args.append("-mcpu=apple-m1")

bruce_c_module = Extension(
    "bruce_c",
    sources=["src/bruce_c/bruce_numpy.c", "src/bruce_c/bruce_src.c"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="bruce",
    version="1.1.0",
    description="A fast-as-hell binary star model using NumPy-C API with OpenMP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[bruce_c_module],

    entry_points={
        "console_scripts": [
            "lcmatch = utils.lcmatch:main",
            "tesstpf = utils.tesstpf:main",
            "spocfit = utils.spocfit:main",
        ],
    },
)