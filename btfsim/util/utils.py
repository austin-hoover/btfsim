import glob
import hashlib
import os
import subprocess
from collections import OrderedDict


def file_to_dict(filename):
    """Writes text file with two columns to dictionary."""
    filestring = open(filename, "U").read().split("\n")
    thisdict = OrderedDict()
    for item in filestring:
        if item:  # if this isn't an empty line
            if item[0] != "#":  # and if not a comment
                try:  # added "try" layer because last item might be a single-space string
                    items = item.split(", ")
                    itemkey = items[0]  # first piece is item "key"
                    itemvalue = items[1:]
                    if len(itemvalue) == 1:  # if length is 1, save as string not list
                        itemvalue = items[1]
                    thisdict[
                        itemkey
                    ] = itemvalue  # other pieces are stored in dictionary
                except:
                    print("Skipped line '%s'" % item)
                    pass
    return thisdict


def get_latest_file(path, *paths):
    """Returns the name of the latest (most recent) file of the joined path(s)"""
    fullpath = os.path.join(path, *paths)
    list_of_files = glob.glob(fullpath)  # You may use iglob in Python3
    if not list_of_files:  # I prefer using the negation
        return None  # because it behaves like a shortcut
    latest_file = max(list_of_files, key=os.path.getctime)
    _, filename = os.path.split(latest_file)
    return filename


def git_url():
    url = None
    try:
        url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode("utf-8").strip()
        if url and url.endswith('.git'):
            url = url[:-4]
    except subprocess.CalledProcessError:
        pass
    return url


def is_git_clean():
    clean = False
    try:
        clean = False if subprocess.check_output(['git', 'status', '--porcelain']) else True
    except subprocess.CalledProcessError:
        pass
    return clean


def git_revision_hash():
    commit = None
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
    except subprocess.CalledProcessError:
        pass
    return str(commit)


def file_hash(filename):
    sha1_hash = hashlib.sha1()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha1_hash.update(byte_block)
    return sha1_hash.hexdigest()


def url_style(url, text=None):
    if text is None:
        text = url
    return {'text': text, 'url': url}