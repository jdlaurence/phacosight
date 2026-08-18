"""Harmonization-map unit tests: synthetic CSVs, no dataset needed."""

import numpy as np
import pandas as pd
import pytest

from phacosight.data.cataracts import (
    IGNORE_INDEX,
    STEP_TO_C1K_ID,
    STEP_TO_PHASE,
    STEPS,
    labels_at_times,
    load_steps,
    to_c1k,
)
from phacosight.phase.timeline import PHASE_ID, PHASES


def test_map_covers_all_steps_and_targets_exist():
    assert set(STEP_TO_PHASE) == set(range(len(STEPS)))
    for name in STEP_TO_PHASE.values():
        assert name is None or name in PHASES


def test_lookup_table_matches_dict():
    assert STEP_TO_C1K_ID.shape == (len(STEPS),)
    assert STEP_TO_C1K_ID[0] == PHASE_ID["idle"]
    assert STEP_TO_C1K_ID[9] == IGNORE_INDEX          # Vitrectomy
    assert STEP_TO_C1K_ID[15] == PHASE_ID["Viscoelastic_Suction"]
    assert STEP_TO_C1K_ID[7] == STEP_TO_C1K_ID[8] == PHASE_ID["Phacoemulsification"]


def test_load_steps_valid(tmp_path):
    csv = tmp_path / "train99.csv"
    pd.DataFrame({"Frame": [1, 2, 3, 4], "Steps": [0, 3, 3, 9]}).to_csv(csv, index=False)
    steps = load_steps(csv)
    assert steps.tolist() == [0, 3, 3, 9]
    mapped = to_c1k(steps)
    assert mapped.tolist() == [PHASE_ID["idle"], PHASE_ID["Incision"],
                               PHASE_ID["Incision"], IGNORE_INDEX]


@pytest.mark.parametrize("frames,steps", [
    ([2, 3, 4], [0, 0, 0]),        # doesn't start at 1
    ([1, 2, 4], [0, 0, 0]),        # gap
    ([1, 2, 3], [0, 0, 19]),       # ID out of range
])
def test_load_steps_rejects_bad_csvs(tmp_path, frames, steps):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"Frame": frames, "Steps": steps}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        load_steps(csv)


def test_labels_at_times_sampling():
    steps = np.array([0] * 30 + [3] * 30 + [9] * 30)   # 3 s @ 30 fps: idle, incision, vitrectomy
    times = np.array([0.0, 1.0, 2.0, 2.9])
    labels = labels_at_times(steps, 30.0, times)
    assert labels.tolist() == [PHASE_ID["idle"], PHASE_ID["Incision"],
                               IGNORE_INDEX, IGNORE_INDEX]
    # out-of-range times clamp to the last frame rather than crashing
    assert labels_at_times(steps, 30.0, np.array([99.0])).tolist() == [IGNORE_INDEX]
