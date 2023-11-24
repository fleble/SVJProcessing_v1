import numba as nb
import numpy as np
import awkward as ak

import utils.builtinUtilities as utl
import utils.awkwardArray.awkwardArrayUtilities as akUtl
from utils.awkwardArray.awkwardArrayWrappers import AkArrayEnhanced


def get_collection_name_from_branch_name(branch_name):
    """Return collection name corresponding to a branch name.

    Args:
        branch_name (str)

    Returns:
        str
    """

    return branch_name.split("_")[0]


def get_n_branch_name_from_collection_name(collection_name):
    """Return the name of the branch storing number of objects for a jagged collection.

    Args:
        collection_name (str)

    Returns:
        str
    """

    return "n" + collection_name


@nb.jit
def make_reindexing_list(filter_row):
    converter = []
    cum_sum = 0
    for x in filter_row:
        converter.append(cum_sum)
        if x: cum_sum += 1
    return converter
 
@nb.jit
def __sum(list_):
    n = 0
    for x in list_: n += x
    return n

@nb.jit
def reindex(builder, filter_, indices, jet_cands_matching_table):
    for indices_row, filter_row in zip(indices, filter_):
        builder.begin_list()
        converter = make_reindexing_list(filter_row)
        if jet_cands_matching_table:
            n = len(filter_row)
        else:
            n = __sum(filter_row)
        for idx in indices_row:
            if idx == -1:
                new_idx = -1
            elif idx >= 0 and idx < n:
                new_idx = converter[idx]
            else:
                new_idx = -1
            builder.append(new_idx)
        builder.end_list()
    return builder
 

class EventsFromAkArray(AkArrayEnhanced):
    """Wrapper of awkward.highlevel.Array to easily filter events and objects.

    Implements the following public methods:
        filter(filter_ak_array)
        filter_collection(collection_name, filter_ak_array)
        get_collection(collection_name)

    By default the filter methods modified the object inplace.
    The get_collection method returns a new EventsFromAkArray with only the
    branches corresponding to the collection.
    """

    def __init__(self, ak_array, pf_cands_collection_name="PFCands"):
        """
        Args:
            ak_array (awkward.Array)
        """

        ak.behavior["EventsFromAkArray"] = EventsFromAkArray
        if "__array__" not in ak.parameters(ak_array):
            ak_array = ak.with_parameter(
                ak_array,
                "__array__",
                "EventsFromAkArray",
            )

        super().__init__(ak_array)

        self.collection_names = []
        self.jet_constituents_matching_table_names = {}
        self.jet_collection_names = []
        self.constituent_collection_names = []
        self.pf_cands_name = pf_cands_collection_name

        self.__set_collection_names()
        self.__set_jet_constituents_matching_table_names()
        self.__set_jet_collection_names()
        self.__set_constituent_collection_names()

    def __set_collection_names(self):
        """Define attribute collection_names."""

        collection_names = []
        for branch_name in self.keys():
            if not "_" in branch_name: continue
            collection_name = branch_name.split("_")[0]
            if collection_name not in collection_names:
                collection_names.append(collection_name)
        self.collection_names = collection_names

    def __set_jet_constituents_matching_table_names(self):
        """Define attribute jet_constituents_matching_table_name.

        Figure out from branches names which jet and constituent collections
        are "bound" together assuming the following convention:
        if branch JetCollectionNameConstituentCollectionName_pFCandsIdx,
        and branch JetCollectionNameConstituentCollectionName_jetIdx exist
        then ConstituentCollectionName are constituents of JetCollectionName
        and the matching is performed by these 2 branches.

        Special hard-coded case for default gen PFNanoAOD collections.
        """

        for field in self.keys():
            if not field.endswith("_pFCandsIdx"): continue
            table_name = field.replace("_pFCandsIdx", "")
            if table_name + "_jetIdx" not in self.keys(): continue

            for idx in range(1, len(table_name)-1):
                jet_collection_name_candidate = table_name[:idx]
                constituent_collection_name_candidate = table_name[idx:]
                found = False
                if (jet_collection_name_candidate in self.collection_names
                    and constituent_collection_name_candidate in self.collection_names):
                    found = True
                    break

            if found:
                jet_collection_name = jet_collection_name_candidate
                constituent_collection_name = constituent_collection_name_candidate
            else:
                if table_name == "GenJetCands":
                    jet_collection_name = "GenJet"
                    constituent_collection_name = "GenCands"
                elif table_name == "GenFatJetCands":
                    jet_collection_name = "GenFatJet"
                    constituent_collection_name = "GenCands"
                else:
                    print("Warning: Will not be able to filter using table %s" % table_name)
                    continue
                    
            key = (jet_collection_name, constituent_collection_name)
            self.jet_constituents_matching_table_names[key] = table_name

    def __set_jet_collection_names(self):
        """Define attribute jet_collection_names."""
        self.jet_collection_names = list(set([x for x, y in self.jet_constituents_matching_table_names.keys()]))

    def __set_constituent_collection_names(self):
        """Define attribute constituent_collection_names."""
        self.constituent_collection_names = list(set([y for x, y in self.jet_constituents_matching_table_names.keys()]))

    def copy(self):
        """Return a deep copy of the array (no memory shared with original)."""
        return EventsFromAkArray(ak.copy(self))

    def filter(self, event_filter, inplace=True):
        """Event filtering.

        Args:
            event_filter (awkward.highlevel.Array):
                A 1D boolean ak array to filter events
            inplace (bool, optional, default=True):
                Modify instance inplace or return a modified copy

        Returns:
            None or EventsFromAkArray
        """

        if inplace:
            self.__init__(self[event_filter])
        else:
            return EventsFromAkArray(self[event_filter])

    def filter_collection(self, collection_name, collection_filter):
        """Object filtering. Modify instance inplace.

        Args:
            collection_name (str): Name of the collection to filter
            collection_filter (awkward.highlevel.Array):
                A 2D boolean ak array to filter objects within events

        Returns:
            None
        """

        n_branch_updated = False
        idx_branches_updated = False

        for field in self.keys():
            field_collection_name = get_collection_name_from_branch_name(field)
            if field_collection_name == collection_name:

                # Filter branch
                new_branch = self[field][collection_filter]
                self[field] = new_branch

                # Update n branch
                if not n_branch_updated:
                    self.__update_n_branch(collection_name, field)
                    n_branch_updated = True

                # Update idx branches
                if not idx_branches_updated:
                    self.__update_idx_branches(collection_name, collection_filter)
                    idx_branches_updated = True

        return

    def get_collection(self, collection_name, remove_collection_name_prefix=True):
        """Return new EventsFromAkArray with only the branches corresponding to the collection.

        Args:
            collection_name (str): Name of the collection for which to return EventsFromAkArray.
            remove_collection_name_prefix (bool, optional, default=True):
                Remove the CollectionName_ prefix from the branch name.
                In this case, an AkArrayEnhanced is returned

        Returns:
            EventsFromAkArray or AkArrayEnhanced
        """

        branch_names = []
        for field in self.keys():
            field_collection_name = get_collection_name_from_branch_name(field)
            if field_collection_name == collection_name:
                branch_names.append(field)

        if remove_collection_name_prefix:
            new_array = AkArrayEnhanced(self[branch_names])
            for branch_name in branch_names:
                new_array.rename_field(branch_name, branch_name.replace(f"{collection_name}_", ""))
            return new_array

        else:
            return EventsFromAkArray(self[branch_names])

    def to_nested_array(self, variables_to_keep, jet_collection_name, n_jets_max=None, n_constituents_max=None):
        """

        n_jets_max and n_constituents_max should be all None or all not None.
        """
    
        assert (n_jets_max is None and n_constituents_max is None) or (n_jets_max is not None and n_constituents_max is not None)

        arrays = {}

        pf_cands = None
        for variable_name in variables_to_keep:
            if variable_name.startswith(self.pf_cands_name+"_"):

                if pf_cands is None:
                    pf_cands_variables = [variable_name for variable_name in variables_to_keep if variable_name.startswith(self.pf_cands_name)]
                    pf_cands = self.__get_pf_cands(jet_collection_name, pf_cands_variables)
                    nan = {k: np.nan for k in pf_cands.fields}
                    if n_jets_max is None:
                        n_jets_max = ak.max(getattr(self, "n" + jet_collection_name), axis=None)
                    pf_cands_per_jet_idx = []
                 
                    for jet_idx in range(n_jets_max):
                        pf_cands_this_jet = pf_cands[pf_cands.jetIdx==jet_idx]
                        # Make a rectangular array if n_constituents_max is not None
                        if n_constituents_max is not None:
                            pf_cands_this_jet = ak.pad_none(pf_cands_this_jet, n_constituents_max, axis=1, clip=True)
                            pf_cands_this_jet = ak.fill_none(pf_cands_this_jet, nan)
                        pf_cands_per_jet_idx.append(akUtl.as_regular(pf_cands_this_jet))

                    pf_cands_per_jet_idx = ak.Array(pf_cands_per_jet_idx)

                variable = pf_cands_per_jet_idx[variable_name]
                variable = akUtl.move_axis(variable, 0, 1)
                if n_jets_max is not None and n_constituents_max is not None:
                    variable = akUtl.as_regular(variable)
                arrays[variable_name] = variable

            elif n_jets_max is not None and variable_name.startswith(jet_collection_name):
                variable = ak.pad_none(self[variable_name], n_jets_max, axis=1, clip=True)
                variable = ak.fill_none(variable, np.nan)
                arrays[variable_name] = variable

            else:
                if variable_name not in self.keys():
                    print(f"Variable {variable_name} not found in file!")
                    exit(0)
                
                arrays[variable_name] = self[variable_name]

        return AkArrayEnhanced(ak.Array(arrays))

    @staticmethod
    def __make_new_indices(filter_, indices, jet_cands_matching_table):
        """Update collection indices after selecting some objects from a collection.
    
        Args
            filter_ (awkward.highlevel.Array): Jagged array with boolean values
                corresponding to whether the object is selected or not
            indices (list[list[int]]): Jagged array with former indices
            jet_cands_matching_table (bool): Reindexing for the jet - PF cands matching table
    
        Returns:
            awkward.highlevel.Array: Jagged array with new indices
        """

        builder = ak.ArrayBuilder()
        return reindex(builder, filter_, indices, jet_cands_matching_table).snapshot()

    def __update_n_branch(self, collection_name, filtered_branch_name):
         """Update n<Object> branch after object selection.

         E.g. update "nJet" after selecting some "Jet" with "Jet_pt > 200".

         Args:
             collection_name (str)
             filtered_branch_name (str)

         Returns:
             None
         """

         n_branch_name = get_n_branch_name_from_collection_name(collection_name)
         new_n_branch = ak.num(self[filtered_branch_name], axis=1)
         self[n_branch_name] = new_n_branch
         return

    def __update_idx_branches(self, collection_name, collection_filter):
        """Update <object>Idx branch after object selection.
 
        E.g. update "Electron_jetIdx" after selecting some "Jet".

        Args:
            collection_name (str)
            collection_filter (awkward.highlevel.Array)

        Returns:
            None
        """

        # jetIdx for PFCands using matching table (Fat)JetPFCands
        # e.g. if selecting some "Jet", then "JetPFCands_jetIdx" must be updated and the
        # JetPFCands collection must be filterd to remove entries corresponding to non selected "Jet"
        if collection_name in self.jet_collection_names:
            for (jet_collection_name, _), matching_table_name in self.jet_constituents_matching_table_names.items():
                if collection_name != jet_collection_name: continue
                idx_branch_name = matching_table_name + "_jetIdx"
                if not hasattr(self, idx_branch_name):
                   print("Cannot update jet - pf cands matching table because it is missing.")
                   return
                # Indices of the selected jets
                old_indices = akUtl.get_true_indices(collection_filter)
                # Boolean ak array to keep only entries in the matching table corresponding to
                # PF candidates clustered in selected jets
                pf_cands_filter = akUtl.is_in(getattr(self, idx_branch_name), old_indices)
                # Filter matching table as described aboved
                self.filter_collection(matching_table_name, pf_cands_filter)
                # Update the indices to selected jets
                self[idx_branch_name] = self.__make_new_indices(collection_filter, self[idx_branch_name], jet_cands_matching_table=True)

        # PF candidates idx: nothing to do, just pass
        elif collection_name in self.jet_constituents_matching_table_names.values():
            pass

        # All other collections
        # e.g. if selecting some "Photon", then "Electron_photonIdx" must be updated
        else:
            for field in self.keys():
                pattern = "_" + collection_name[0].lower() + collection_name[1:] + "Idx$"
                if utl.in_regex(field, [pattern]):
                    self[field] = self.__make_new_indices(collection_filter, self[field], jet_cands_matching_table=False)

        return

    def __get_pf_cands(self, jet_collection_name, variables_to_keep=None):
        """Get jet pf candidates collection for a given jet collection.

        Args:
            events (awkward.highlevel.Array): the Events TTree opened with uproot.
            jet_collection_name (str): Jet or FatJet
            variables (list[str] or None): list of variables to keep.
                If None, all variables are, else e.g. ["PFCands_pt"]

        Returns:
            awkward.highlevel.Array: ak array with field pt, eta, phi, mass,
                charge and pdgId, jetIdx and possibly other fields like
                ptOverJetPt, deltaEta, deltaPhi, deltaR.
        """

        is_none = variables_to_keep is None
        jet_pf_cands_collection_name = jet_collection_name + self.pf_cands_name

        jet_idx = eval("self." + jet_collection_name + "PFCands_jetIdx")

        # Branches to keep
        if variables_to_keep is None:
            variables_to_keep = [ field for field in self.fields if field.startswith(self.pf_cands_name + "_") ]

        variables = [variable for variable in variables_to_keep if variable in self.fields]
        if len(variables) > 0:
            pf_cands = ak.zip({variable: self[variable] for variable in variables_to_keep if variable in self.fields})

            # Select only PF candidates clustered in jets
            jet_pf_cands_idx = eval("self." + jet_pf_cands_collection_name + "_pFCandsIdx")
            pf_cands = pf_cands[jet_pf_cands_idx]

            # Add idx information to do the matching with jets
            pf_cands = ak.with_field(pf_cands, jet_idx, where="jetIdx")

        else:
            pf_cands = ak.zip({"jetIdx": jet_idx})

        # Add other variables
        if is_none:
            jet_pf_cands_variable_names = [
                field for field in self.fields
                if (field.startswith(jet_pf_cands_collection_name + "_")
                    and field != jet_pf_cands_collection_name + "_pFCandsIdx"
                    and field != jet_pf_cands_collection_name + "_jetIdx")
            ]
        else:
            variables = [v.replace(self.pf_cands_name + "_", jet_pf_cands_collection_name + "_") for v in variables_to_keep]
            jet_pf_cands_variable_names = [
                field for field in self.fields
                if (field.startswith(jet_pf_cands_collection_name + "_")
                    and field != jet_pf_cands_collection_name + "_pFCandsIdx"
                    and field != jet_pf_cands_collection_name + "_jetIdx"
                    and field in variables
                   )
            ]

        for collection_variable_name in jet_pf_cands_variable_names:
            variable_name = self.pf_cands_name + collection_variable_name.replace(jet_pf_cands_collection_name, "")
            variable = eval("self." + collection_variable_name)
            pf_cands = ak.with_field(pf_cands, variable, where=variable_name)

        return pf_cands
