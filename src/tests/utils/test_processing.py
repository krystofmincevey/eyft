import pandas as pd
from pandas.testing import assert_frame_equal

from ...eyft.utils.processing import process_str, merge


def test_process_str():
    mapping = {
        'â\x82¬': ' ',
        'â': ' ',
        '¬': ' ',
        'Â': ''
    }
    str_input = "   V.ÂG.  is ¬beautifulâ\x82¬"

    actual_output = process_str(str_input, mapping=mapping)
    expected_output = "V.G. is beautiful"
    assert actual_output == expected_output


def test_merge(left_merge_input, right_merge_input):
    df_actual = merge(
        left_merge_input, right_merge_input,
        left_key='Key', right_key='Key',
        merge_type="left",
    )

    df_expected = pd.DataFrame(
        data=[
            ['a', 10, 1],
            ['b', 20, None],
            [None, 30, None],
        ], columns=['Key', 'Price', 'EPC'],
    )

    assert_frame_equal(df_actual, df_expected, check_dtype=False)
