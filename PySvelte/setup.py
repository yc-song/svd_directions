from setuptools import setup, find_packages

setup(
    name='pysvelte',
    version="1.0.1",
    packages=find_packages(),
    python_requires='>=3.6.0',
    install_requires=[
        'torch',
        'einops',
        'numpy',
        "typeguard",    
    ]
)