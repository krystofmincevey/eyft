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

- [ ] min\_max_scale
- [ ] z_normalise
- [X] mode\_impute; TODO: Add a description.
- [ ] mean\_impute
- [ ] median\_impute
- [ ] categorize
- [ ] cap
- [ ] cap\_3std
- [ ] floor
- [ ] dummy_var
- [ ] geolocate *More aspirational, as it would entail converting 
an address to x,y geo-coordinates using geopandas*
- [ ] pass (i.e. do\_nothing)

**Testing progress**
- [ ] min\_max_scale
- [X] z_normalise; TODO: resolve github issues
- [ ] mode\_impute; TODO: Add a description.
- [ ] mean\_impute
- [ ] median\_impute
- [ ] categorize
- [ ] cap
- [ ] cap\_3std
- [ ] floor
- [ ] dummy_var
- [ ] geolocate *More aspirational, as it would entail converting 
an address to x,y geo-coordinates using geopandas*
- [ ] pass (i.e. do\_nothing)
