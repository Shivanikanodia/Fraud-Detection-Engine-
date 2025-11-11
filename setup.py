from setuptools import setup, find_packages


setup(
name = "ml_toolkit",
version="0.1.0",
author="Shivani",
author_email="kanodiashivani27@gmail.com"
 packages=find_packages(),
install_requires=[
            "numpy>=1.18.0",
            "pandas>=1.0.0",
            "scikit-learn>=0.22.0",
            "matplotlib>=3.2.0",
            flask>=1.1.0,
            joblib>=0.14.0
        ],
      
)