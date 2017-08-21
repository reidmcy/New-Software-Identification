import setuptools
import re

with open('simple_id/__init__.py') as f:
    versionString = re.search(r"__version__ = '(.+)'", f.read()).group(1)

if __name__ == '__main__':
    setuptools.setup(name='simple_id',
        version = versionString,
        author="Reid McIlroy-Young",
        packages = ['simple_id'],
    )
