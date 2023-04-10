# Data Processing:

Preprocessing will be split into two phases.
Firstly, a bespoke phase which will be executed in a jupyter notebook
using functions from the utils directory. And secondly an
automated phase which will perform automated checks
in a kedro pipeline. The params specified in
*./base/parameters/data_processing.yml*
will solely apply to the automated kedro phase.
A user will be expected to specify a dictionary of
col: modification_type pairs. These will dictate which
columns are selected by the rest of the engine, and what
processing is applied to the various columns. Eg. Min-Max Scaling,
Mean Var normalisation, etc.

**WARNING:** The modifications will be specified as a series of
strings in a list. The names specified will have to match
processing methods supported by the engine. Please specify what 
preprocessing functions are supported below.

- [ ] boxcox_normalise
- [X] cap
- [X] cap_3std
- [ ] cat_dummies
- [ ] categorize
- [X] pass (i.e. select without processing)
- [X] floor
- [X] floor_and_cap
- [X] mean_impute
- [X] median_impute
- [X] min_max_scale
- [X] mode_impute
- [X] segment
- [X] z_normalise


**To obtain a list of all implemented functions** simply execute the 
following code in the python console:
    
    import inspect
    
    # Import the module
    from src.eyft.pipelines.data_processing import processor as mymodule
    
    # Get a list of all names defined in the module
    names = dir(mymodule)
    
    # Iterate over the names and check if each is a function
    for name in names:
        # Get the object for the name
        obj = getattr(mymodule, name)
        # Check if the object is a function
        if inspect.isfunction(obj):
            print(name)

