from setuptools import setup

VERSION = 'v0.1.0'

readme_name = "readme.md"
with open(readme_name) as fh:
    long_description = fh.read()

setup(
    name="pygenopt",
    version=VERSION,
    description="A generic framework for writing linear optimization problems in Python.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords=["Modeling Framework", "Linear Programming", "Linear Optimization", "Operations Research"],
    author="Guilherme Freitas Coelho",
    author_email="coelhoxz@gmail.com",
    url="https://github.com/guifcoelho/pygenopt",
    python_requires=">=3.10",
    packages=[
        "pygenopt", "pygenopt.solvers",
    ],
    include_package_data=True,
    install_requires=[],
)