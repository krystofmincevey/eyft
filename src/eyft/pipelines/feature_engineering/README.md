# Feature Engineering:

Following in the footsteps of data processing, 
the feature engineering pipeline 
permits users to specify what transformations
(including multiplication and division by other features)
they would like to apply to the processed df.
Note that the engineered features are added to 
the feature space, and do not replace existing features.
The transformations are specified as a dictionary
where the key dictates the column used 
to generate the new features and the values 
indicates how the new features are to be created.

Implement:
- [ ] log
- [ ] divide\_by
- [ ] multiply\_by
- [ ] nan\_categorize
- [ ] geolocate *More aspirational, as it would entail converting 
an address to x,y geo-coordinates using geopandas*
- ...