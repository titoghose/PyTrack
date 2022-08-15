import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTrack-NTU",
    version="1.1.1",
    author="Upamanyu Ghose, Arvind Srinivasan",
    author_email="titoghose@gmail.com, 96arvind@gmail.com",
    description="An end-to-end python analysis toolkit for eye tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/titoghose/PyTrack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "scipy", "matplotlib", 
        "pandas", "Pillow>=5.4.0", "sqlalchemy>=1.2.15", 
        "statsmodels", "pingouin==0.2.2"]
)