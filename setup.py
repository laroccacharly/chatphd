from setuptools import setup, find_packages

setup(
    name="chatphd",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "streamlit",
        "pydantic",
    ],
    python_requires=">=3.9",
    author="Charly Robinson La Rocca",
    description="Chat with academic papers using Claude",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 