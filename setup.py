import os
import sys
import misp
from setuptools import setup


# "setup.py publish" shortcut.
if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist')
    os.system('twine upload dist/*')
    os.system('rm -rf dist misp.egg-info')
    sys.exit()

install_requires = ['numpy', 'matplotlib', 'scikit-learn', 'tqdm', 'openslide-python',
                    'torch==1.0.1', 'torchvision==0.2.2']

setup(
    name='misp',
    version=misp.__version__,
    description="Medical Image Segmentation Pipeline",
    url='https://github.com/zhoudaxia233/misp',
    license='MIT',
    packages=['misp'],
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    python_requires='>=3.6',
    dependency_links=[]
)