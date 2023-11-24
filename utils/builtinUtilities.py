import os
import re

from utils.Logger import *


def in_regex(name, regex_list):
    """
    Returns the list of the indices of the regexes which the text matches.
    An empty list is returned if the text matches none of the regexes.

    Args:
        name (str)
        regex_list (list): list or any type that can be casted to a list
    
    Returns:
        list[int]
    """

    if not isinstance(regex_list, list):
        regex_list = [regex_list]
    indices = []
    for idx, regex in enumerate(regex_list):
        research = re.search(regex, name)
        if (research != None):
           indices.append(idx)

    return indices


def in_regex_list(name_list, regex_list):
    """
    Returns the list of the indices of the regexes which the text matches.
    False is returned if the list of text match none of the regexes.

    Args:
        name_list (list[str])
        regex_list (list): list or any type that can be casted to a list
    
    Returns:
        list[list[int]]
    """

    indices = []
    found = False
    for name in name_list:
        indices_this_name = in_regex(name, regex_list)
        indices.append(indices_this_name)
        found += (indices_this_name != [])

    if isinstance(regex_list, str):
        indices = [i for i, x in enumerate(indices) if len(x) == 1]

    if not found:
        return False
    else:
        return indices


def glob_re(pattern, directory="", prepend_directory=True):
    """Return filenames matching a certain pattern.

    Args: 
        pattern (str): Pattern to search
        directory (str): Directory from which to search files
    
    Returns:
        list[str]: List of filenames matching pattern
    """
    
    file_names = []
    for f in os.listdir(directory):
        if re.search(pattern, f) and os.path.isfile(os.path.join(directory, f)):
            file_name = os.path.join(directory, f) if prepend_directory else f
            file_names.append(file_name)

    return file_names 


def capitalize(word):
    return word[0].upper() + word[1:]
