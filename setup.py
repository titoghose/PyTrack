import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTrack-NTU",
    version="0.0.13",
    author="Upamanyu Ghose, Arvind Srinivasan",
    author_email="titoghose@gmail.com, 96arvind@gmail.com",
    description="An end-to-end python analysis toolkit for eye tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pytrack-ntu.readthedocs.io",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy>=1.16.2", "scipy==1.2.1", "matplotlib>=3.0.2", "pandas", "Pillow", "sqlalchemy", "statsmodels", "pingouin"]
)