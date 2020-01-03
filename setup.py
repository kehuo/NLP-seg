from setuptools import setup, find_packages
import os
import inspect

__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))

def get_install_requires(path):
    content = open(os.path.join(__location__, path)).read()
    return [req for req in content.split('\\n') if req != '']

setup(
    name="NLP-seg",
    version="1.3.2.3",
    py_modules=["nlp_seg"],
    author="BaseBit",
    description="Deep Learning tagging tools",
    packages=["prep", "post", "model", "med_parser", "med_tagger"],
    # packages=find_packages(),
    include_package_data=True,
    platforms='any',
    install_requires=get_install_requires('requirements.txt'),
    data_files=[
        ('', ['requirements.txt', 'setup.py', 'MANIFEST.in'])
    ],
)
