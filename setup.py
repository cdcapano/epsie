import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="epsie",
    version="0.0.1",
    author="Collin D. Capano",
    author_email="cdcapano@gmail.com",
    description="EPSIE is an Embarrassingly Parallel Sampler for "
                "Inference Estimation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cdcapano/epsie",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
)
