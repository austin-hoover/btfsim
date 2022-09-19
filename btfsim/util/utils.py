import glob
import os
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
