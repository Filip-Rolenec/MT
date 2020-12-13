from progressbar import progressbar

from gas_example.optimization.basis_function import get_basis_functions
from gas_example.setup import GasProblemSetup
from gas_example.optimization.vf import create_vfs_time_list, update_vf_coef


def adp_algorithm_complete():
    problem_setup = GasProblemSetup()

    vfs = create_vfs_time_list(problem_setup.time_epochs)
    basis_functions = get_basis_functions()

    for time_epoch in progressbar(reversed(problem_setup.time_epochs)):
        # Value function in the last time epoch is always zero,
        # there are no money to be gained, we do not update it.
        if time_epoch != len(problem_setup.time_epochs) - 1:
            # print(f"{datetime.now()} time epoch: {time_epoch}")
            update_vf_coef(vfs[time_epoch], vfs[time_epoch + 1], basis_functions, time_epoch)

    return vfs
