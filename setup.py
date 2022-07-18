from setuptools import setup

setup (
    name="pyCydonia",
    version="0.1",
    packages=['pyCydonia.reader', 'pyCydonia.profiler', 'pyCydonia.dataProcessor'],
    install_requires=["numpy", "pandas", "asserts", "argparse"]
)
