import os
import re

from setuptools import setup, find_packages

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',  # It excludes inline comment too
    open("si/__init__.py").read(),
).group(1)

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()

setup(
    name="PSI",
    version=__version__,
    author="Tran Tuan Kiet, Nguyen Thang Loi, Duong Tan Loc, Vo Nguyen Le Duy, PSI Contributors",
    long_description=README,
    long_description_content_type="text/markdown",
    author_email="contact.trtkiet@gmail.com; duyvnl@uit.edu.vn",
    url="https://github.com/PythonSI/PSI",
    packages=find_packages(),
    install_requires=[
        "matplotlib==3.10.6",
        "mpmath==1.3.0",
        "numpy==2.2.6",
        "POT==0.9.5",
        "scikit-learn==1.7.1",
        "scipy==1.15.3",
    ],
    python_requires=">=3.10",
)
