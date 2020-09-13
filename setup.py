import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="enal",  # Required
    version="1.0.0",  # Required
    description="Energy trading analytics toolkit",  # Optional
    package_dir={"": "src"},  # Optional
    packages=find_packages(where="src"),  # Required
    python_requires=">=3.6",
    # install_requires=['psycopg2', 'pandas'],  # Optional
)
