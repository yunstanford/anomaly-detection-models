import os
import subprocess
from uranium import task_requires


def main(build):
    build.packages.install(".", develop=True)


def distribute(build):
    """ distribute the uranium package """
    build.packages.install("wheel")
    build.executables.run([
        "python", "setup.py",
        "sdist", "bdist_wheel", "--universal", "upload"
    ])
