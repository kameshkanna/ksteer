from setuptools import find_packages, setup

setup(
    name="ksteer",
    version="0.1.0",
    description="Calibrated activation steering for behavioral alignment",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.27.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
    ],
)
