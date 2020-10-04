from progressbar import progressbar

from simple_example.basis_function import get_basis_functions
from simple_example.setup import ProblemSetup
from simple_example.vf import create_vfs, update_vf_coef


def adp_algorithm_final(loops_of_update: int,
                        sample_size: int,
                        problem_setup: ProblemSetup
                        ):
    vfs = create_vfs(problem_setup.time_epochs, [0, 0, 0])
    basis_functions = get_basis_functions()

    for loop in progressbar(range(loops_of_update)):
        for time_epoch in reversed(problem_setup.time_epochs):
            if time_epoch != len(problem_setup.time_epochs) - 1:
                update_vf_coef(vfs[time_epoch], vfs[time_epoch + 1], problem_setup, sample_size, basis_functions)

    return vfs
