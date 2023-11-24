import argparse
from importlib import import_module

import awkward as ak
from coffea import processor, nanoevents
from coffea.nanoevents.methods import vector

import utils.uprootUtilities as uprootUtl
from utils.eventsWrapper import EventsFromAkArray
from utils.coffea.akArrayAccumulator import AkArrayAccumulator
from skimmer import skimmerUtils

# Needed so that ak.zip({"pt": [...], "eta": [...], "phi": [...], "mass": [...]},
#                         with_name="PtEtaPhiMLorentzVector")
# is understood as a PtEtaPhiMLorentzVector from coffea.nanoevents.methods.vector
ak.behavior.update(vector.behavior)


class Skimmer(processor.ProcessorABC):

    def __init__(self, process_function):
        self.accumulator = AkArrayAccumulator()
        self.converted_branches = []
        self.process_function = process_function

        
    def __get_collection_names(self, tree_marker_events):
        collection_candidates = {}
        for branch_name in tree_marker_events.fields:
            if "_" in branch_name:
                collection_candidate = branch_name.split("_")[0]
            elif "fCoordinates" in branch_name:
                collection_candidate = branch_name.split("/")[0]
            else:
                continue

            if collection_candidate not in collection_candidates.keys():
                collection_candidates[collection_candidate] = 1
            else:
                collection_candidates[collection_candidate] += 1

        exceptions = (
              [f"{x}JECdown" for x in collection_candidates]
            + [f"{x}JECup" for x in collection_candidates]
            + [f"{x}JERdown" for x in collection_candidates]
            + [f"{x}JERup" for x in collection_candidates]
        )
        collection_names = [x for x, n in collection_candidates.items() if n > 1 and x not in exceptions]

        return collection_names


    def __convert_collection_to_nano_aod(self, tree_maker_events, collection_name):

        self.converted_branches.append(collection_name)

        tree_maker_collection_branch_names = []
        jet_constituents_table_branch_names = []

        # Default branches
        for branch_name in tree_maker_events.fields:
            if not branch_name.startswith(f"{collection_name}_"): continue
            if branch_name.startswith(f"{collection_name}_subjets"): continue
            if branch_name.endswith("_constituentsIndex") or branch_name.endswith("_constituentsIndexCounts"):
                jet_constituents_table_branch_names.append(branch_name)
            else:
                tree_maker_collection_branch_names.append(branch_name)
        
        collection = {
            b: tree_maker_events[b] for b in tree_maker_collection_branch_names
        }
        self.converted_branches += tree_maker_collection_branch_names

        # pT, eta, phi, mass/energy branches
        tree_maker_four_vector_variables = ["Pt", "Eta", "Phi", "E"]
        lorentz_four_vector_variables = ["pt", "eta", "phi", "energy"]
        four_vector_dict = {}
        for tm_variable, l_variable in zip(tree_maker_four_vector_variables, lorentz_four_vector_variables):
            branch_name = f"{collection_name}/{collection_name}.fCoordinates.f{tm_variable}"
            four_vector_dict[l_variable] = tree_maker_events[branch_name]
            self.converted_branches.append(branch_name)
        four_vector = ak.zip(four_vector_dict, with_name="PtEtaPhiELorentzVector")

        collection[f"{collection_name}_pt"] = four_vector.pt
        collection[f"{collection_name}_eta"] = four_vector.eta
        collection[f"{collection_name}_phi"] = four_vector.phi
        collection[f"{collection_name}_mass"] = four_vector.mass

        # Systematics branches
        systematics_branch_names = [
            f"{collection_name}JECdown_jerFactor",
            f"{collection_name}JECdown_origIndex",
            f"{collection_name}JECup_jerFactor",
            f"{collection_name}JECup_origIndex",
            f"{collection_name}JERdown_origIndex",
            f"{collection_name}JERup_origIndex",
        ]
        output_systematics_branch_names = [
            f"{collection_name}_JECdownJerFactor",
            f"{collection_name}_JECdownOrigIndex",
            f"{collection_name}_JECupJerFactor",
            f"{collection_name}_JECupOrigIndex",
            f"{collection_name}_JERdownOrigIndex",
            f"{collection_name}_JERupOrigIndex",
        ]
        for output_branch_name, systematics_branch_name in zip(
            systematics_branch_names,
            output_systematics_branch_names,
        ):
            if systematics_branch_name in tree_maker_events.fields:
                self.converted_branches.append(systematics_branch_name)
                collection[output_branch_name] = tree_maker_events[systematics_branch_name]

        # Subjet branches
        subjet_count_branch_name = f"{collection_name}_subjetsCounts"
        if subjet_count_branch_name in tree_maker_events.fields:
            self.converted_branches.append(subjet_count_branch_name)
            for branch_name in tree_maker_events.fields:
                if branch_name.startswith(f"{collection_name}_subjets/{collection_name}_subjets.fCoordinates"):
                    self.converted_branches.append(branch_name)
                    # The branches will not appear in the outpu file
                elif branch_name.startswith(f"{collection_name}_subjets_"):
                    self.converted_branches.append(branch_name)
                    # The branches will not appear in the outpu file
            

        # Jet to constituents matching tables
        jet_constituents_table = {}
        if len(jet_constituents_table_branch_names) == 2:
            input_branch_name = f"{collection_name}_constituentsIndex"
            output_branch_name = f"{collection_name}JetsConstituents_pFCandsIdx"
            jet_constituents_table[output_branch_name] = tree_maker_events[input_branch_name]
            self.converted_branches.append(input_branch_name)

            input_branch_name = f"{collection_name}_constituentsIndexCounts"
            output_branch_name = f"{collection_name}JetsConstituents_jetIdx"
            # TODO: JIT this
            branch = []
            for event in tree_maker_events[input_branch_name]:
                branch.append([])
                for idx, count in enumerate(event):
                    branch[-1] += count * [idx]
            jet_constituents_table[output_branch_name] = ak.Array(branch)
            self.converted_branches.append(input_branch_name)

        collection = {k: ak.copy(v) for k, v in collection.items()}
        jet_constituents_table = {k: ak.copy(v) for k, v in jet_constituents_table.items()}

        return collection, jet_constituents_table


    def __convert_non_collection_branches(self, tree_maker_events):
        branches = {}
        for branch_name in tree_maker_events.fields:
            if branch_name not in self.converted_branches:
                branches[branch_name] = ak.copy(tree_maker_events[branch_name])
        
        return branches


    def __convert_to_nano_aod(self, tree_maker_events):

        nano_events = {}

        collection_names = self.__get_collection_names(tree_maker_events)

        self.converted_branches = []
        for collection_name in collection_names:
            collection, jet_constituents_table = self.__convert_collection_to_nano_aod(tree_maker_events, collection_name)
            nano_events.update(collection)
            nano_events.update(jet_constituents_table)

        non_collection_branches = self.__convert_non_collection_branches(tree_maker_events)
        nano_events.update(non_collection_branches)
            
        return nano_events


    def process(self, tree_maker_events):

        nano_events = self.__convert_to_nano_aod(tree_maker_events)
        nano_events = EventsFromAkArray(ak.Array(nano_events))

        cut_flow = {}
        skimmerUtils.update_cut_flow(cut_flow, "Initial", nano_events)

        nano_events, cut_flow = self.process_function(nano_events, cut_flow)
        skimmerUtils.update_cut_flow(cut_flow, "Final", nano_events)

        accumulator = {
            "events": AkArrayAccumulator(nano_events),
            "cut_flow": cut_flow,
        }

        return accumulator


    def postprocess(self, accumulator):
        return super().postprocess(accumulator)


def __get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_file_names",
        help="Input file names",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_file_name",
        help="Output file name",
        required=True,
    )
    parser.add_argument(
        "-m", "--process_module_name",
        help="Process module name, e.g. analysisConfigs.config",
        required=True,
    )
    parser.add_argument(
        "-y", "--year",
        help="Data-taking year",
        required=True,
    )

    return parser.parse_args()


def main():

    args = __get_arguments()
    input_file_names = args.input_file_names.split(",")
    process_module = import_module(args.process_module_name)
    process_function = lambda x, y: process_module.process(x, y, year=args.year)

    # TODO: replace by read arguments
    executor = processor.iterative_executor
    executor_args = {
        "schema": nanoevents.BaseSchema,
        "workers": 1,
    }
    chunk_size = 400
    max_chunks = None

    # Calculate new branches
    accumulator = processor.run_uproot_job(
        {"fileset": input_file_names},
        treename="TreeMaker2/PreSelection",
        processor_instance=Skimmer(process_function),
        executor=executor,
        executor_args=executor_args,
        chunksize=chunk_size,
        maxchunks=max_chunks,
        )

    trees = {}

    # Making output ROOT file
    trees["Events"] = accumulator["events"].value
    trees["CutFlow"] = accumulator["cut_flow"]

    uprootUtl.write_root_file(
        output_file_name=args.output_file_name,
        trees=trees,
    )


if __name__ == "__main__":
    main()
