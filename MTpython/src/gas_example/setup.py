from gas_example.enum_types import Action

TIME_EPOCHS = 300

GAS_PRICE = 23
GAS_VOL = 0.04

CO2_PRICE = 9
CO2_VOL = 0.025

POWER_PRICE = 40
POWER_VOL = 0.06

GOV_PROB_UP = 0.08
GOV_PROB_DOWN = 0.04

RISK_FREE_RATE = 0.02
BORROW_RATE = 0.07


class GasProblemSetup:

    def __init__(self):
        self.time_epochs = range(TIME_EPOCHS)
        # self.actions =
        self.init_gas_price: float = GAS_PRICE
        self.gas_vol: float = GAS_VOL
        self.init_co2_price: float = CO2_PRICE
        self.co2_vol: float = CO2_VOL
        self.init_power_price: float = POWER_PRICE
        self.power_vol: float = POWER_VOL
        self.gov_prob_up: float = GOV_PROB_UP
        self.gov_prob_down: float = GOV_PROB_DOWN
        self.epoch_rf_rate: float = get_epoch_rate(RISK_FREE_RATE)
        self.epoch_b_rate: float = get_epoch_rate(BORROW_RATE)

# When epochs are months
def get_epoch_rate(year_rate):
    return year_rate**(1/float(12))