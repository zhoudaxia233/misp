import os
import sys
import platform
import misp
from setuptools import setup, find_packages


# "setup.py publish" shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist')
    os.system('twine upload dist/*')
    if platform.system() == 'Windows':
        os.system('powershell rm –path dist, misp.egg-info –recurse –force')
    else:
        os.system('rm -rf dist misp.egg-info')
    sys.exit()

install_requires = ['numpy>=1.16.0', 'matplotlib>=3.0.1', 'scikit-learn>=0.20.2',
                    'tqdm>=4.19.9', 'openslide-python>=1.1.1', 'torch>=1.1.0', 'torchvision>=0.2.2']

setup(
    name='misp',
    version=misp.__version__,
    description="Medical Image Segmentation Pipeline",
    url='https://github.com/zhoudaxia233/misp',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    python_requires='>=3.6',
    dependency_links=['https://openslide.org/download/']
)
