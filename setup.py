from setuptools import setup

package_name = 'cgpdm_lib'

setup(
    name=package_name,
    version='1.0',
    packages=[package_name],
    data_files=[],
    install_requires=['torch',
                      'numpy',
                      'matplotlib',
                      'scikit-learn',
                      'termcolor'],
    zip_safe=True,
    maintainer='fabio-amadio',
    maintainer_email='fabioamadio93@gmail.com',
    description='GPDM and CGPDM PyTorch-based implementation',
    license='GNU GENERAL PUBLIC LICENSE v3',
    entry_points={},
)