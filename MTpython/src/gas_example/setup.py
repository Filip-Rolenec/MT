from gas_example.enum_types import Action

TIME_EPOCHS = 300
class GasProblemSetup:

    def __init__(self):
        self.time_epochs = range(TIME_EPOCHS)
        self.actions = [action for action in Action]
        self.states = range(3)
