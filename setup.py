import os
import re

from setuptools import setup, find_packages

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open("pythonsi/__init__.py").read(),
).group(1)

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()

setup(
    name="pyselectinf",
    version=__version__,
    author="Tran Tuan Kiet, Nguyen Thang Loi, Duong Tan Loc, Vo Nguyen Le Duy, PythonSI Contributors",
    long_description=README,
    long_description_content_type="text/markdown",
    author_email="contact.trtkiet@gmail.com; duyvnl@uit.edu.vn",
    url="https://github.com/PythonSI/PySelectInf",
    packages=find_packages(),
    install_requires=[
        "mpmath==1.3.0",
        "numpy==2.2.6",
        "POT==0.9.5",
        "scikit-learn==1.7.1",
        "scipy==1.15.3",
    ],
    python_requires=">=3.10",
)
