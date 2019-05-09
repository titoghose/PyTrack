import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTrack-NTU",
    version="0.0.2",
    author="Upamanyu Ghose, Arvind Srinivasan",
    author_email="titoghose@gmail.com, 96arvind@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/titoghose/PyTrack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy==1.16.2",
        "scipy==1.2.1",
        "pingouin==0.2.2",
        "sqlalchemy",
        "pandas==0.23.4",
        "matplotlib==3.0.2",
        "statsmodels==0.9.0",
        "Pillow"
      ]
)