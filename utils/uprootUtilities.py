import re
from functools import reduce

import numpy as np
import awkward as ak
import uproot

import utils.awkwardArray.awkwardArrayUtilities as akUtl
from utils.awkwardArray.awkwardArrayWrappers import AkArrayEnhanced
from utils.Logger import *


def get_keys_from_file(file_):
    """Return keys of a ROOT file open with uproot4 without suffix ;1.

    Args:
        file_ (uproot.writing.writable.WritableDirectory)

    Returns:
        list[str]
    """

    return [key.split(";")[0] for key in file_.keys()]


def write_tree_to_root_file(file_, tree_name, data):
    """Write histograms, stored in a coffea accumulator, to a ROOT file.

    Args:
        file_ (uproot.writing.writable.WritableDirectory)
        tree_name (str)
        data (ak.Array, dict or coffea.processor.accumulator.dict_accumulator)

    Returns:
        None
    """
    
    ak_array = AkArrayEnhanced(akUtl.to_ak_array(data))
    sort_by_name = tree_name != "CutFlow"
    branches = __make_branches(ak_array, sort_by_name)

    if tree_name not in get_keys_from_file(file_):
        file_[tree_name] = branches
    else:
        file_[tree_name].extend(branches)

    return


def write_root_file(output_file_name, mode="recreate", trees={}):
    """Write ROOT file with Events and Metadata TTrees.

    Args:
        output_file_name (str): Full path to store the file.
        mode (str): "recreate" or "update"
        trees (dict[str, any]):
            Keys are tree name
            Values are tree content

    Returns:
        None
    """

    log.blank_line()
    log.info("Writing down output ROOT file %s" % output_file_name)
    with getattr(uproot, mode)(output_file_name) as output_file:
        for tree_name, tree in trees.items():
            write_tree_to_root_file(output_file, tree_name, tree)
            log.info("TTree %s saved to output file" % tree_name)

    return


def __get_collection_and_variable_names(field):
    """Get collection and variable names from field name.

    Args:
        field (str)

    Returns:
        tuple(str, str)
    """

    split = field.split("_")
    if len(split) > 2:
        return split[0], reduce(lambda x, y: f"{x}_{y}", split[1:])
    if len(split) == 2:
        return split[0], split[1]
    else:
        return split[0], None


def __get_jagged_collections(ak_array):
    """Get collections in the ak array.

    Args:
       ak_array (awkward.Array): ak array with fields

    Returns:
        list[str]
    """

    collections = []
    for field in ak_array.fields:
        collection_candidate, variable = __get_collection_and_variable_names(field)
        if variable is None or collection_candidate is None: continue
        if not "var" in str(ak.type(ak_array[field])): continue
        if len(ak_array[field]) == 1: continue
        if collection_candidate not in collections:
            collections.append(collection_candidate)

    return collections
        

def __cast_unknown_type_branches_to_float(ak_array):

    fields = ak_array.fields
    for field in fields:
        type_text = str(ak.type(ak_array[field]))
        if "unknown" in type_text:
            log.warning(f"Branch {field} has unknown type, converting to float64")
            branch = akUtl.as_type(ak_array.pop(field), np.float64)
            ak_array.add_field(field, branch)
        elif "?" in type_text:
            new_type_text = type_text.split("*")[-1].replace("?", "").replace(" ", "")
            log.warning(f"Branch {field} has unclear type {type_text}, converting to {new_type_text}")
            branch = akUtl.as_type(ak_array.pop(field), getattr(np, new_type_text))
            ak_array.add_field(field, branch)


def __make_branches(ak_array, sort_by_name=True):
    """Get object that can be written as a TTree to a ROOT file with uproot4.

    Args:
       ak_array (awkward.Array): ak array with fields

    Returns:
        list[str]
    """

    __cast_unknown_type_branches_to_float(ak_array)
    collections = __get_jagged_collections(ak_array)

    # Book dictionaries for arrays to zip together and that does not not have to be zipped
    branches_to_zip = {collection: {} for collection in collections}
    single_branches = {}

    # Fill in those dictionaries
    for field in ak_array.fields:
        is_collection_variable = False
        
        collection_candidate, variable = __get_collection_and_variable_names(field)
        for collection in collections:
            if collection_candidate == collection:
                branches_to_zip[collection][variable] = ak_array[field]
                is_collection_variable = True

        if not is_collection_variable:
            single_branches[field] = ak_array[field]

    # Zip branches and add everything in a dict
    collections_branches = {}
    for collection in branches_to_zip.keys():
        try:
            collections_branches[collection] = ak.zip(branches_to_zip[collection])
        except ValueError:
            log.warning(f"Inconsistent sizes of branches for collection {collection}")
            log.warning("Collection will be skipped.")
            for branch_name, branch in branches_to_zip[collection].items():
                log.warning("%s: %d (%d)" % (branch_name, ak.count(branch, axis=None), ak.count(ak.fill_none(branch, 0), axis=None)))

    branches = collections_branches
    if sort_by_name:
        single_branches_names = sorted(single_branches)
    else:
        single_branches_names = single_branches.keys()
    for branch_name in single_branches_names:
        if branch_name not in branches.keys():
            branches[branch_name] = single_branches[branch_name]

    return branches

