EPOCHS_IN_YEAR = 12
YEARS = 25
TIME_EPOCHS = 25 * EPOCHS_IN_YEAR

HOURS_IN_EPOCH = 365 * 24 / EPOCHS_IN_YEAR

POWERPLANT_COST = 65_000_000
MAINTENANCE_COST_PER_MW = 6

GAS_VOL = 0.12

CO2_VOL = 0.10

POWER_VOL = 0.15

RISK_FREE_RATE_YEAR = 1 + 0.02
BORROW_RATE_YEAR = 1 + 0.07

RISK_FREE_RATE_EPOCH = RISK_FREE_RATE_YEAR ** (1 / EPOCHS_IN_YEAR)
BORROW_RATE_EPOCH = BORROW_RATE_YEAR ** (1 / EPOCHS_IN_YEAR)

SAMPLE_SIZE_INDIVIDUAL = 500
SAMPLE_SIZE_GLOBAL = 100
INTEGRAL_SAMPLE_SIZE = 3000


def get_epoch_rate(year_rate):
    return year_rate ** (1 / float(EPOCHS_IN_YEAR))
