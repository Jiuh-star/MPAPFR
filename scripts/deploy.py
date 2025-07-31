from __future__ import annotations

import os
from pathlib import PurePosixPath, WindowsPath

import paramiko

HOST = os.environ["REMOTE_HOST"]
USERNAME = os.environ["USERNAME"]
PASSWORD = os.environ["PASSWORD"]
TARGET = os.environ["TARGET"]
PORT = int(os.environ.get("PORT", 22))
SOURCES = [
    "configs",
    "src",
    "scripts",
    "pyproject.toml",
    "README.md",
]
EXCEPTS = [
    "pfl_attack.egg-info",
    "__pycache__",
]


def main():
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(HOST, username=USERNAME, password=PASSWORD, port=PORT)
    sftp = ssh.open_sftp()

    for source in SOURCES:
        put_file(sftp, WindowsPath(source), PurePosixPath(TARGET) / source)


def put_file(sftp: paramiko.SFTPClient, source: WindowsPath, target: PurePosixPath):
    if source.is_dir():
        try:
            sftp.mkdir(str(target))
        except IOError:
            pass

        for file in source.iterdir():
            put_file(sftp, file, target / file.name)
    else:
        if any(except_ in str(source) for except_ in EXCEPTS):
            return

        print(f"Uploading {source} to {target}")
        sftp.put(str(source), str(target))


if __name__ == "__main__":
    main()
