from setuptools import setup, find_packages

setup(
    name="frte",
    version="0.1.0",
    description="FRTE: First Return, Then Explore — A Non-Markovian Meta-Optimizer for Deep Learning and Non-Convex Optimization",
    author="Emanuel Samir",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "matplotlib"
    ],
    python_requires=">=3.9",
)
