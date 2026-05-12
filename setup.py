from setuptools import find_packages, setup


setup(
    name="article-xas-xncd",
    version="0.1.0",
    description="Base project for XAS/XNCD analysis workflows",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
