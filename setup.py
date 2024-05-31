from setuptools import setup, find_packages

setup(
    name='QSLE',
    version='0.0.6.2',
    author='QRH',
    author_email='Append@gmail.com',
    description='QSLE Toolbox',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numba',
    ],
    classifiers=[
        # Classifiers help users find your project on PyPI
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
