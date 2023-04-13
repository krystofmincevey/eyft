import pytest
import pandas as pd


# @pytest.fixture
 def fs_inputs():

      data = [
          [1.4, 0.9, 0, 5, 70, 0, 20],
          [1.1, 0.9, -10, -5, 140, 10, 80],
          [0.9, 0.001, 100, 15, 60, 20, 45],
          [-1.01, -1.1, 30, -45, -90, 10, 5],
          [-1.10, -1,5, 45, -55, 10, -5, 15],
          [],
          [1],
     ]
     columns = [
         'Price',  # y_col
        'COL1', 'COL2', 'COL3', "COL4", "COL5", "COL6", "COL7"
     ]

     return pd.DataFrame(data, columns=columns)

# TESTING PRINCIPLES
# y = mx1 + m2x2 + c + error
#
# choose some m   m1 = 1.1   m2 = 1.5
#
# x3, x4, x5, x6   RANDOM NUMBERS so should not be selected
#
# USE x1, x2, .... x6 as inputs and y as target.
# FROM ABOVE YOU SHOULD KNOW THAT x1 and x2 get selected.

