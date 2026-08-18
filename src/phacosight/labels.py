"""Class-ID conventions for Cataract-1K segmentation tasks.

These mirror the mask-generation scripts in
``Dataset_codes/semantic segmentation dataset codes/`` exactly — mask PNGs
store the train ID as the pixel value, so these tables are the single source
of truth for interpreting them.

Note the two naming schemes in the upstream scripts: the anatomy scripts match
Supervisely titles like ``'Slit Knife'``, while the multiclass-instrument
"rev" script matches short lowercase titles (``'slit knife'``, ``'rhexis'``).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegTask:
    name: str
    class_names: tuple[str, ...]  # index == train ID
    mask_dir: str                 # per-case mask folder produced by the upstream scripts

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


# Task 1 variant A: anatomy + lumped instruments (json_to_network_mask_anatomy_instrument.py)
ANATOMY_INSTRUMENT = SegTask(
    name="anatomy_instrument",
    class_names=("background", "cornea", "pupil", "lens", "instrument"),
    mask_dir="mask_anatomy_inst",
)

# Task 1 variant B: binary instruments (json_to_network_mask_instrument.py)
INSTRUMENT_BINARY = SegTask(
    name="instrument_binary",
    class_names=("background", "instrument"),
    mask_dir="mask_instruments",
)

# Task 1 variant C: multi-class instruments, 10 tools grouped into 6 classes
# (json_to_network_mask_instrument_classification_rev.py, instrument_dictionary)
INSTRUMENT_MULTICLASS = SegTask(
    name="instrument_multiclass",
    class_names=(
        "background",
        "slit/incision knife",     # 'slit knife', 'incision knife'
        "gauge/cannula/spatula",   # 'gauge', 'rhexis', 'spatula'
        "phacoemulsifier tip",     # 'phaco tip'
        "irrigation-aspiration",   # 'IA'
        "lens injector",           # 'cartridge'
        "forceps",                 # 'katena forceps', 'rhexis forceps'
    ),
    mask_dir="mask_instruments_MultiClass",
)

TASKS = {t.name: t for t in (ANATOMY_INSTRUMENT, INSTRUMENT_BINARY, INSTRUMENT_MULTICLASS)}

# Classes whose IoU gates downstream irregularity work (tracked separately in eval).
PUPIL_CLASS_NAME = "pupil"
