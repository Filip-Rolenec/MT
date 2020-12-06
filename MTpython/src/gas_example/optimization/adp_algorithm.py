from datetime import datetime

from progressbar import progressbar

from gas_example.optimization.model import AdpModel
from gas_example.setup import GasProblemSetup
from gas_example.optimization.vf import create_vfs, update_vf_coef


def adp_algorithm_complete(loops_of_update: int,
                           problem_setup: GasProblemSetup,
                           adp_model: AdpModel):
    vfs = create_vfs(adp_model, problem_setup.time_epochs)

    basis_functions = adp_model.basis_functions

    for _ in progressbar(range(loops_of_update)):
        for time_epoch in reversed(problem_setup.time_epochs):
            if time_epoch != len(problem_setup.time_epochs) - 1:
                print(f"{datetime.now()} time epoch: {time_epoch}")
                time_epoch_left = len(problem_setup.time_epochs) - time_epoch
                update_vf_coef(vfs[time_epoch], vfs[time_epoch + 1], basis_functions, time_epoch, time_epoch_left)

    return vfs
