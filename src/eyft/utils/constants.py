# TODO; Move kwarg defaults into constants.py.

# GENERIC CONSTANTS -------------------------------------------------
SEED = 42


# PROCESSING CONSTANTS ----------------------------------------------


# ENGINEERING CONSTANTS ---------------------------------------------


# SELECTION CONSTANTS -----------------------------------------------
RF_PARAMS = {
    'eta': 0.039,
    'verbosity': 0,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.9,
    'eval_metric': 'mae',
    'random_state': SEED,
}


# MODELLING CONSTANTS -----------------------------------------------


# MODELLING CONSTANTS -----------------------------------------------