import numba as nb
import awkward as ak
from coffea.nanoevents.methods import vector

import utils.awkwardArray.awkwardArrayUtilities as akUtl
import utils.PtEtaPhiMLorentzVectorUtilities as vecUtl
from utils.Logger import *


ak.behavior.update(vector.behavior)


def update_cut_flow(cut_flow, cut_name, events):
    """Update cut flow table in a coffea accumulator.

    Args:
        cut_flow (dict[str, float])
        cut_name (str): the name of the cut to appear in the cut flow tree
        events (EventsFromAkArray)
        use_raw_events (bool)
    """

    if cut_name in cut_flow.keys():
        cut_flow[cut_name] += get_number_of_events(events)
    else:
        cut_flow[cut_name] = get_number_of_events(events)
    return


def get_number_of_events(events):
    return ak.sum(events.Weight)


def apply_trigger_cut(events, trigger_list):
    """Filter events using an or of all triggers.

    Args:
        events (EventsFromAkArray)
        trigger_list (list[str])

    Returns:
        None
    """

    trigger_filter = __get_trigger_filter(events, trigger_list)
    events.filter(trigger_filter, inplace=True)

    return


def apply_met_filters_cut(events, cut_flow, met_filter_names):
    """MET filters cuts.

    Args:
        events (EventsFromAkArray)
        cut_flow (dict[str, float])
        met_filter_names (list[str])

    Returns:
        None
    """

    for met_filter_name in met_filter_names:
        branch_name = "Flag_" + met_filter_name
        if branch_name in events.fields:
            met_filter = getattr(events, branch_name)
            events.filter(met_filter, inplace=True)

    return


def apply_phi_spike_filter(events, year, jet_eta_branch_name="Jet_eta", jet_phi_branch_name="Jet_phi", reverse=False):
    rad = 0.028816 # half the length of the diagonal of the eta-phi rectangular cell
    rad *= 0.35 # the factor of 0.35 was optimized from the signal vs. background sensitivity study

    eta_lead = None
    eta_sub = None
    phi_lead = None
    phi_sub = None
    if year == "2016":
        eta_lead = [0.048,0.24,1.488,1.584,-1.008]
        phi_lead = [-0.35,-0.35,-0.77,-0.77,-1.61]
        eta_sub = [-1.2,-0.912,-0.912,-0.816,-0.72,-0.72,-0.528,-0.432,-0.336,-0.24,-0.24,-0.144,-0.144,-0.048,0.144,0.912,0.912,1.008,1.296,-1.584,-0.816,-0.72,-0.144,-0.048,-0.048,0.048,1.104,1.488]
        phi_sub = [-1.19,2.03,3.01,-1.75,-2.17,-0.77,2.73,2.73,0.21,0.07,0.21,-2.59,0.77,0.91,1.75,1.75,2.87,0.63,-0.49,0.63,1.47,-2.31,0.07,-2.59,0.77,0.91,-3.15,2.73]
    elif year == "2017":
        eta_lead = [0.144,1.488,1.488,1.584,-0.624]
        phi_lead = [-0.35,-0.77,-0.63,-0.77,0.91]
        eta_sub = [-0.912,-0.912,-0.816,-0.72,-0.528,-0.336,-0.24,-0.24,-0.144,-0.144,-0.048,0.144,0.912,0.912,1.008,-1.2,-0.72,-0.72,-0.432,0.336,0.624,1.104,1.296]
        phi_sub = [2.03,3.01,-1.75,-0.77,2.73,0.21,0.07,0.21,-2.59,0.77,0.91,1.75,1.75,2.87,0.63,-1.19,-2.31,-2.17,2.73,-0.77,-0.77,-3.15,-0.49]
    elif year == "2018":
        eta_lead = [1.488,1.488,1.584]
        phi_lead = [-0.77,-0.63,-0.77]
        eta_sub = [-1.584,-1.2,-0.912,-0.912,-0.816,-0.816,-0.72,-0.72,-0.528,-0.432,-0.336,-0.24,-0.24,-0.144,-0.144,-0.144,-0.048,-0.048,0.144,0.912,0.912,1.008,1.296,-0.72,1.104,1.488,1.776]
        phi_sub = [0.63,-1.19,2.03,3.01,-1.75,-0.77,-2.17,-0.77,2.73,2.73,0.21,0.07,0.21,-2.59,0.07,0.77,0.77,0.91,1.75,1.75,2.87,0.63,-0.49,-2.31,-3.15,-0.21,0.77]
    else:
        raise ValueError("Invalid year")

    eta_lead = nb.typed.List(eta_lead)
    eta_sub = nb.typed.List(eta_sub)
    phi_lead = nb.typed.List(phi_lead)
    phi_sub = nb.typed.List(phi_sub)

    jets_eta = getattr(events, jet_eta_branch_name)
    jets_phi = getattr(events, jet_phi_branch_name)

    builder = ak.ArrayBuilder()
    phi_spike_filter = __get_phi_spike_filter(builder, eta_lead, phi_lead, eta_sub, phi_sub, rad, jets_eta, jets_phi, reverse=reverse).snapshot()

    events.filter(phi_spike_filter, inplace=True)

    return


def apply_hem_filter(events, year, jet_pt_branch_name="Jet_pt", jet_eta_branch_name="Jet_eta",
                     jet_phi_branch_name="Jet_phi"):

    hem_min_pt = 30
    hem_min_eta = -3.05
    hem_max_eta = -1.35
    hem_min_phi = -1.62
    hem_max_phi = -0.82

    if year != "2018":
        hem_filter = ak.ones_like(events.genWeight)
    else:
        jets_pt = getattr(events, jet_pt_branch_name)
        jets_eta = getattr(events, jet_eta_branch_name)
        jets_phi = getattr(events, jet_phi_branch_name)
        
        condition_per_jet = (jets_pt > hem_min_pt) \
            & (jets_eta > hem_min_eta) & (jets_eta < hem_max_eta) \
            & (jets_phi > hem_min_phi) & (jets_phi < hem_max_phi)
        hem_filter = ak.any(condition_per_jet, axis=1) == False
    events.filter(hem_filter, inplace=True)

    return


def apply_jet_cleaning(
        events,
        jet_collection_name,
        cleaning_collection_name=None,
        cleaning_collection=None,
        no_cleaning_condition=None,
        radius=None,
    ):
    """Apply jet cleaning.

    Args:
        events (EventsFromAkArray)
        jet_collection_name (str): e.g. "Jet" or "FatJet"
        cleaning_collection_name (str): e.g. "electron" or "muon"
        radius (float): cleaning radius. For `jet_collection_name`
            "Jet" or "FatJet" will be automatically inferred if None.
        no_cleaning_condition (ak.Array, optional, default=None):
            Do not remove the jet if it matches this condition. If None, simple
            delta R cleaning is applied.
    """

    if radius is None:
        if jet_collection_name == "Jet" and radius is None:
            radius = 0.4
        elif jet_collection_name == "FatJet" and radius is None:
            radius = 0.8
        elif radius is None:
            log.critical(f"No default radius for jet collection {jet_collection_name}")

    variables_4vector = ["pt", "eta", "phi", "mass"]

    jet_collection = events.get_collection(jet_collection_name)
    jets_4vector = ak.zip(
        {variable: jet_collection[variable] for variable in variables_4vector},
        with_name="PtEtaPhiMLorentzVector",
    )

    if cleaning_collection is None:
        cleaning_collection = events.get_collection(cleaning_collection_name)
    cleaning_collection_4vector = ak.zip(
        {variable: cleaning_collection[variable] for variable in variables_4vector},
        with_name="PtEtaPhiMLorentzVector",
    )

    jets_, cleaning_collection_ = ak.unzip(ak.cartesian([jets_4vector, cleaning_collection_4vector], axis=-1, nested=True))
    min_delta_r = ak.min(vecUtl.delta_r(jets_, cleaning_collection_), axis=-1)
    # If an event has 0 object in the jet or cleaning collections
    # Then this results in None, and the jet should not be removed, replacing delta R by 2 radius
    # TODO: replace by this line once moving to awkward 2
    #min_delta_r = ak.drop_none(min_delta_r)
    min_delta_r = ak.fill_none(min_delta_r, 2*radius)
    dtype = str(ak.type(min_delta_r)).split("*")[-1].replace("?", "").replace(" ", "")
    min_delta_r = akUtl.as_type(min_delta_r, dtype)  # needed because otherwise the type is messed up

    cleaning_filter = (min_delta_r > radius)
    if no_cleaning_condition is not None:
        cleaning_filter = cleaning_filter | no_cleaning_condition

    events.filter_collection(jet_collection_name, cleaning_filter)


def collections_matching(
        events,
        collection1_name=None,
        collection2_name=None,
        collection1=None,
        collection2=None,
    ):
    """Return the mapping of closest object in collection1 to each object of collection2.
    
    Use as:
    mapping = collections_matching(events, collection1_name, collection2_name)
    matched_collection2 = collection2[mapping]

    The matching is a simple closest delta R with repetition!
    There can be several times the same object from collection2 matched.

    Either `collection1_name` is None and `collection1` is not None, or 
    vice-versa. Same for `collection2_name` and `collection2`.

    Args:
        events (EventsFromAkArray)
        collection1_name (str)
        collection2_name (str)
        collection1 (ak.Array): ak array with fields "pt", "eta", "phi", "mass"
        collection2 (ak.Array): ak array with fields "pt", "eta", "phi", "mass" 
    """

    variables_4vector = ["pt", "eta", "phi", "mass"]

    if collection1 is None:
        collection1 = events.get_collection(collection1_name)
    collection1_4vector = ak.zip(
        {variable: collection1[variable] for variable in variables_4vector},
        with_name="PtEtaPhiMLorentzVector",
    )

    if collection2 is None:
        collection2 = events.get_collection(collection2_name)
    collection2_4vector = ak.zip(
        {variable: collection2[variable] for variable in variables_4vector},
        with_name="PtEtaPhiMLorentzVector",
    )
    
    collection1_, collection2_ = ak.unzip(ak.cartesian([collection1_4vector, collection2_4vector], axis=-1, nested=True))
    mapping = ak.argmin(vecUtl.delta_r(collection1_, collection2_), axis=-1)
    dtype = str(ak.type(mapping)).split("*")[-1].replace("?", "").replace(" ", "")
    mapping = akUtl.as_type(mapping, dtype)  # needed because otherwise the type is messed up

    return mapping


def __get_trigger_filter(events, trigger_list):
    """Return ak array corresponding to an or of the triggers in input list.

    Args:
        events (awkward.highlevel.Array): or object inheriting from ak.Array
        trigger_list (list[str])

    Returns:
        awkward.highlevel.Array
    """

    trigger_filter = None
    for trigger in trigger_list:
        trigger_branch = getattr(events, "HLT_" + trigger)
        if trigger_filter is None:
            trigger_filter = trigger_branch
        else:
            trigger_filter = (trigger_filter | trigger_branch)

    return trigger_filter


@nb.jit
def __get_phi_spike_filter(builder, eta_lead, phi_lead, eta_sub, phi_sub, rad, jets_eta, jets_phi, reverse):
    for jet_eta, jet_phi in zip(jets_eta, jets_phi):
        if len(jet_eta) < 2:
            builder.append(True)
        else:
            keep_event = True
            for iep in range(len(eta_lead)):
                if (eta_lead[iep] - jet_eta[0])**2 + (phi_lead[iep] - jet_phi[0])**2 < rad:
                    keep_event = False
                    break
            for iep in range(len(eta_sub)):
                if (eta_sub[iep] - jet_eta[1])**2 + (phi_sub[iep] - jet_phi[1])**2 < rad:
                    keep_event = False
                    break
            if reverse:
                builder.append(not keep_event)
            else:
                builder.append(keep_event)
    return builder

