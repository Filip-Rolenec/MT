from progressbar import progressbar

from gas_example.optimization_2.vf import create_vfs_time_list, update_vf_coef
from gas_example.setup import TIME_EPOCHS


def adp_algorithm_complete():
    vfs = create_vfs_time_list()

    for time_epoch_left in progressbar(range(TIME_EPOCHS)):
        time_epoch = TIME_EPOCHS - time_epoch_left-1
        # Last Vf value is zero, no money to be gained.
        if time_epoch != TIME_EPOCHS - 1:
            update_vf_coef(vfs[time_epoch], vfs[time_epoch + 1], time_epoch)
    return vfs
