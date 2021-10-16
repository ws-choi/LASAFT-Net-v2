from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    REQUIREMENTS = f.readlines()

setup(
    name='lasaft',
    version='0.0.1',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Topic :: Artistic Software',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Editors',
        'Topic :: Software Development :: Libraries',
    ],
    description='LASAFT-Net-v2',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Woosung Choi',
    author_email='ws_choi@korea.ac.kr',
    license='MIT',
    packages=find_packages(),
    keywords=['audio', 'source', 'separation', 'music', 'sound', 'source separation', 'musdb18', 'mdx'],
    install_requires=[
        REQUIREMENTS
    ],
    include_package_data=True
)
