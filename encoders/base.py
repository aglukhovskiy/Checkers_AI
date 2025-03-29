
import importlib

def get_encoder_by_name(name):  # <1>
    module = importlib.import_module('encoders.' + name)
    constructor = getattr(module, 'create')  # <3>
    return constructor()
