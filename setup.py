import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="model_builder",
    version="0.2.1",
    author="Wilmer Zwietering",
    author_email="wilmer@zwietering.com",
    description="Easily build Keras models from code or from a configuration file",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/wzwietering/model_builder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Utilities",
    ],
    install_requires=["tensorflow"],
    python_requires=">=3.6",
)
