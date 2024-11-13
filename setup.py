from setuptools import setup, find_packages

setup(
    name="chatphd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aiohappyeyeballs>=2.4.3",
        "aiohttp>=3.10.10",
        "anthropic>=0.39.0",
        "streamlit>=1.40.1",
        "pyyaml>=6.0.2",
    ],
    python_requires=">=3.9",
    author="Charly Robinson La Rocca",
    description="Chat with academic papers using Claude",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 