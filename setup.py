import os
from pathlib import Path
from typing import List

import setuptools

_PATH_ROOT = Path(os.path.dirname(__file__))


def load_requirements(filename: str, comment_char: str = "#") -> List[str]:
    """Load requirements from text file in the requirements directory."""
    with (_PATH_ROOT / "requirements" / filename).open() as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filter all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


def load_version() -> str:
    """Load version from the lightly/__init__.py file.

    Note: We do not want to get the version by accessing `lightly.__version__` because
    it would require importing `lightly`. Importing `lightly` in setup.py breaks the
    installation process as the import has side effects and requires dependencies to be
    installed. As dependencies are not yet available during installation, the `lightly`
    import fails.
    """
    version_filepath = _PATH_ROOT / "lightly" / "__init__.py"
    with version_filepath.open() as file:
        for line in file.readlines():
            if line.startswith("__version__"):
                version = line.split("=")[-1].strip().strip('"')
                return version
    raise RuntimeError("Unable to find version string in '{version_filepath}'.")


if __name__ == "__main__":
    name = "lightly"
    version = load_version()
    author = "Lightly Team"
    author_email = "team@lightly.ai"
    description = "A deep learning package for self-supervised learning"
    long_description = (_PATH_ROOT / "README.md").read_text()

    entry_points = {
        "console_scripts": [
            "lightly-crop = lightly.cli.crop_cli:entry",
            "lightly-download = lightly.cli.download_cli:entry",
            "lightly-embed = lightly.cli.embed_cli:entry",
            "lightly-magic = lightly.cli.lightly_cli:entry",
            "lightly-serve = lightly.cli.serve_cli:entry",
            "lightly-train = lightly.cli.train_cli:entry",
            "lightly-version = lightly.cli.version_cli:entry",
        ]
    }

    python_requires = ">=3.6"
    base_requires = load_requirements(filename="base.txt")
    openapi_requires = load_requirements(filename="openapi.txt")
    torch_requires = load_requirements(filename="torch.txt")
    video_requires = load_requirements(filename="video.txt")
    dev_requires = load_requirements(filename="dev.txt")

    setup_requires = ["setuptools>=21"]
    install_requires = base_requires + openapi_requires + torch_requires
    extras_require = {
        "video": video_requires,
        "dev": dev_requires,
        "all": dev_requires + video_requires,
    }

    packages = setuptools.find_packages(include=["lightly*"])

    project_urls = {
        "Homepage": "https://www.lightly.ai",
        "Web-App": "https://app.lightly.ai",
        "Documentation": "https://docs.lightly.ai",
        "Github": "https://github.com/lightly-ai/lightly",
        "Discord": "https://discord.gg/xvNJW94",
    }

    classifiers = [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ]

    setuptools.setup(
        name=name,
        version=version,
        author=author,
        author_email=author_email,
        description=description,
        entry_points=entry_points,
        license="MIT",
        license_files=["LICENSE.txt"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require=extras_require,
        python_requires=python_requires,
        packages=packages,
        classifiers=classifiers,
        include_package_data=True,
        project_urls=project_urls,
    )
