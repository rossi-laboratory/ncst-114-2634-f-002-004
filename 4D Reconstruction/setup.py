"""Setup script for dynamic-4d-reconstruction."""
from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="dynamic-4d-reconstruction",
    version="0.1.0",
    description="4D dynamic scene reconstruction from monocular videos via 3D point tracking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NSTC Frontier AI Lab",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "notebooks", "docs")),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "timm>=0.9.12",
        "open3d>=0.17.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.16.0",
        "tqdm>=4.66.0",
        "einops>=0.7.0",
        "PyYAML>=6.0",
        "trimesh>=4.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
