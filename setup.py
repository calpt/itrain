from setuptools import setup


setup(
    name="itrain",
    version="0.0",
    author="calpt",
    author_email="calpt@mail.de",
    description="Simple training setup",
    license="MIT",
    py_modules=["itrain"],
    install_requires=[
        "adapter-transformers >= 3.0.0",
        "datasets",
        "nltk",
        "scikit-learn",
        "seqeval == 1.2.2",
        "tgsend >= 0.3",
        "torch >= 1.11.0",
        "tqdm",
        "yagmail",
        "keyring",
        "wandb",
        "ruamel.yaml",
    ],
    entry_points={
        "console_scripts": ["itrain=itrain.cli:main"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
