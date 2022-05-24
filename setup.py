import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='neural_tangents_extension',
    version='1.0',
    author='Anonymous Author',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Adapt the Neural Tangents library to differential equation solvers',
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
)