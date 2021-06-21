import setuptools
import sys
import os

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

PATH_ROOT = PATH_ROOT = os.path.dirname(__file__)
builtins.__LIGHTLY_SETUP__ = True

import lightly

def load_description(path_dir=PATH_ROOT, filename='DOCS.md'):
    """Load long description from readme in the path_dir/ directory

    """
    with open(os.path.join(path_dir, filename)) as f:
        long_description = f.read()
    return long_description


def load_requirements(path_dir=PATH_ROOT, filename='base.txt', comment_char='#'):
    """From pytorch-lightning repo: https://github.com/PyTorchLightning/pytorch-lightning.
       Load requirements from text file in the path_dir/requirements/ directory.

    """
    with open(os.path.join(path_dir, 'requirements', filename), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


if __name__ == '__main__':
    
    name = 'lightly'
    version = lightly.__version__
    description = lightly.__doc__

    author = 'Philipp Wirth & Igor Susmelj'
    author_email = 'philipp@lightly.ai'
    description = "A deep learning package for self-supervised learning"

    entry_points = {
        "console_scripts": [
            "lightly-crop = lightly.cli.crop_cli:entry",
            "lightly-train = lightly.cli.train_cli:entry",
            "lightly-embed = lightly.cli.embed_cli:entry",
            "lightly-magic = lightly.cli.lightly_cli:entry",
            "lightly-upload = lightly.cli.upload_cli:entry",
            "lightly-download = lightly.cli.download_cli:entry",
            "lightly-version = lightly.cli.version_cli:entry",
        ]
    }

    long_description = load_description()

    python_requires = '>=3.6'
    install_requires = load_requirements()
    video_requires = load_requirements(filename='video.txt')
    dev_requires = load_requirements(filename='dev.txt')
    all_requires = dev_requires + video_requires
    extras_require = {
        'video': video_requires,
        'dev': dev_requires,
        'all': all_requires,
    }

    packages = [
        'lightly',
        'lightly.api',
        'lightly.cli',
        'lightly.cli.config',
        'lightly.cropping',
        'lightly.data',
        'lightly.embedding',
        'lightly.loss',
        'lightly.loss.regularizer',
        'lightly.models',
        'lightly.models.modules',
        'lightly.transforms',
        'lightly.utils',
        'lightly.active_learning',
        'lightly.active_learning.agents',
        'lightly.active_learning.config',
        'lightly.active_learning.scorers',
        'lightly.active_learning.utils',
        'lightly.openapi_generated',
        'lightly.openapi_generated.swagger_client',
        'lightly.openapi_generated.swagger_client.api',
        'lightly.openapi_generated.swagger_client.models'
    ]

    project_urls = {
        'Homepage': 'https://www.lightly.ai',
        'Web-App': 'https://app.lightly.ai',
        'Documentation': 'https://docs.lightly.ai',
        'Github': 'https://github.com/lightly-ai/lightly',
        'Discord': 'https://discord.gg/xvNJW94',
    }

    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License"
    ]

    setuptools.setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        entry_points=entry_points,
        license='MIT',
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=python_requires,
        packages=packages,
        classifiers=classifiers,
        include_package_data=True,
        project_urls=project_urls,
    )


