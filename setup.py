from setuptools import setup, find_packages

with open("README.md", "r") as rm:
    long_description = rm.read()

setup(
    name="LenSimu",
    version="0.0.1",
    author="Axel Guinot",
    author_email="axel.guinot.astro@gmail.com",
    description="Image simulation for lensing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aguinot/LenSimu",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
