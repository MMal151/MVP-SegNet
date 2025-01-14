from setuptools import setup, find_packages

setup(
    name="mvp-segnet",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[],
    author="Mishaim Malik",
    author_email="mmal151@aucklanduni.ac.nz",
    description="A deep learning model for stroke lesion segmentation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MMal151/MVP-SegNet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    #Dummy commit
)
