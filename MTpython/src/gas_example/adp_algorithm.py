from gas_example.model import AdpModel
from gas_example.setup import GasProblemSetup
from gas_example.vf import create_vfs


def adp_algorithm_final(loops_of_update: int,
                        sample_size: int,
                        problem_setup: GasProblemSetup,
                        adp_model: AdpModel):
    vfs = create_vfs(problem_setup.time_epochs, adp_model.number_of_parameters)
    basis_functions = get_basis_functions()

    for loop in progressbar(range(loops_of_update)):
        for time_epoch in reversed(problem_setup.time_epochs):
            if time_epoch != len(problem_setup.time_epochs) - 1:
                update_vf_coef(vfs[time_epoch], vfs[time_epoch + 1], problem_setup, sample_size, basis_functions)

    return vfs
