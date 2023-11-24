import awkward as ak

from skimmer import skimmerUtils


def process(events, cut_flow, year):
    """Example pre-selection config."""

    # Objects filter
    # For jet ID, see https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL
    # For electron ID, see https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
    # For muon ID, see https://twiki.cern.ch/twiki/bin/viewauth/CMS/SWGuideMuonIdRun2
    events.filter_collection("Jets", (events.Jets_pt > 30) & (abs(events.Jets_eta) < 2.4) & (events.Jets_ID == 1))
    events.filter_collection("JetsAK8", (abs(events.JetsAK8_eta) < 2.4) & (events.JetsAK8_ID == 1))

    # ST cut
    st = events.MET + ak.sum(events.JetsAK8_pt, axis=1)
    events.filter(st > 1300)
    skimmerUtils.update_cut_flow(cut_flow, "STGt1300GeV", events)

    return events, cut_flow

