import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="csfd",
    version="0.0.1",
    author="ranchantan",
    author_email="propella@example.com",
    description="FPE",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com",
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
    include_package_data=True
)
