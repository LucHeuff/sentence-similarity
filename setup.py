from setuptools import setup

setup(
    name="sentence-similarity",
    version="1.0.1",
    description="Algorithm for fast comparison of string sentences",
    url="https://github.com/LucHeuff/sentence-similarity.git",
    author="Luc Heuff",
    author_email="lucheuff@hotmail.com",
    license="MIT",
    packages=["sentences_similarity"],
    install_requires=["numpy", "pandas", "strsimpy"],
)

