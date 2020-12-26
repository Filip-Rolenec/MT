from enum import Enum

class Action(Enum):
    DO_NOTHING = 0  # Do not run the plant and also do not build the new stage.
    RUN = 1  # Run what exists,
    RUN_AND_BUILD = 2  # Run what exists and build a new stage
    IDLE_AND_BUILD = 3  # Do not run and build a new stage


class PowerplantState(Enum):
    NOT_BUILT = 0
    STAGE_1 = 1
    STAGE_2 = 2

