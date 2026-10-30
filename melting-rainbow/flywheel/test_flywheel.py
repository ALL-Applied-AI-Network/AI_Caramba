"""Quick checks for the flywheel (no torch, no dataset needed):  python flywheel/test_flywheel.py"""
import argparse, os, sys, tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from flywheel.examples import save_example, load_examples   # noqa: E402


def frame(v):
    return np.full((480, 640, 3), v, np.uint8)


def test_save_and_load():
    with tempfile.TemporaryDirectory() as d:
        vial, ref = dict(frame=frame(100), ml=70.0), dict(frame=frame(200), ml=70.0)
        assert save_example(d, "run1", 4, 6, "B2", "REJECT", "REVIEW", vial, ref, dict(dE_ref=8.1, ai_p_off=0.3))
        assert save_example(d, "run1", 5, 6, "B2", "PASS", "REJECT", vial, ref)       # operator overrules a PULL
        assert save_example(d, "run1", 4, 6, "B2", "PASS", "REVIEW", vial, ref)       # changes mind on vial 4
        assert save_example(d, "run1", 4, 6, "B2", "", "REVIEW", vial, ref) is None   # not a decision
        rows = {r["vial"]: r for r in load_examples([d])}                           # operator_output folder
        assert set(rows) == {"4", "5"}
        assert rows["4"]["label"] == "good" and rows["4"]["source"] == "review"      # last click wins
        assert rows["5"]["label"] == "good" and rows["5"]["source"] == "override"
        assert os.path.exists(rows["4"]["vial_image"]) and os.path.exists(rows["4"]["ref_image"])
        assert len(os.listdir(os.path.join(d, "flywheel", "images"))) == 3            # reference saved once


def test_station_click_saves_example():
    """A Pass / Reject click in the station writes an example; the station keeps working if saving fails."""
    from operator_ui.operator_app import Engine
    with tempfile.TemporaryDirectory() as d:
        args = argparse.Namespace(ai_model=None, video=None, camera="0", plan=None, batch_size=3, tol=7.0,
                                  speed=8, out=d, no_examples=False, detector="level")
        e = Engine(args)
        ins = e.ins
        for k in range(3):
            ins.vials.append(dict(n=k + 1, frame=frame(50 + k), ml=70.0))
            ins.batch_of[k] = "B1"
        ins.batches = [[0, 1, 2]]
        ins.results = {0: dict(is_reference=False, verdict="OFF-COLOR", dE_ref=12.0),
                       1: dict(is_reference=False, verdict="GOOD", dE_ref=2.0),
                       2: dict(is_reference=True, verdict="REFERENCE", dE_ref=0.0)}
        e.decide(vial=1, action="REJECT")      # station said PULL (rule only, AI off) -> confirm
        e.decide(vial=2, action="REJECT")      # station said PASS -> override
        e.decide(vial=2, action="PASS")        # changed mind: still compared with what the STATION said
        e.decide(vial=3, action="PASS")        # reference vial: not an example
        rows = {r["vial"]: r for r in load_examples([d])}
        assert set(rows) == {"1", "2"}
        assert (rows["1"]["label"], rows["1"]["source"]) == ("off", "confirm")
        assert (rows["2"]["label"], rows["2"]["source"], rows["2"]["station_status"]) == ("good", "confirm", "PASS")
        ins.vials[1]["frame"] = "not an image"                     # saving fails -> logged, no exception
        e.decide(vial=2, action="REJECT")
        assert "example_error" in open(os.path.join(d, "operator_log.csv")).read()
        e.args.no_examples = True                                  # --no-examples: nothing saved
        n = len(open(os.path.join(d, "flywheel", "examples.csv")).readlines())
        e.decide(vial=1, action="PASS")
        assert len(open(os.path.join(d, "flywheel", "examples.csv")).readlines()) == n


if __name__ == "__main__":
    test_save_and_load()
    test_station_click_saves_example()
    print("flywheel tests passed")
