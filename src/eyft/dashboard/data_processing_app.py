import base64
import io
import sys
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import urllib

from typing import Union

from dash import dcc, html, dash_table
from dash_extensions.enrich import (
    DashProxy, MultiplexerTransform,
    Input, Output, State,
)

from src.eyft.pipelines.data_processing.processor import (
    boxcox_normalise, cap, cap_3std, cat_dummies,
    categorize, floor, floor_and_cap, mean_impute,
    median_impute, min_max_scale, mode_impute,
    segment, z_normalise,
)
from src.eyft.pipelines.feature_engineering.transform import (
    log_transform, inverse, multiply_by, divide_by
)


EXTERNAL_STYLESHEETS = [dbc.themes.BOOTSTRAP]

PROCESSING_OPTIONS = [
    {'label': 'Mean Imputation', 'value': 'mean_impute'},
    {'label': 'Median Imputation', 'value': 'median_impute'},
    {'label': 'Mode Imputation', 'value': 'mode_impute'},
    {'label': 'Min-Max Scaling', 'value': 'min_max_scale'},
    {'label': 'Z-Normalization', 'value': 'z_normalise'},
    {'label': 'Outlier Removal: 3 STD', 'value': 'cap_3std'},
    {'label': 'Floor: 1prc', 'value': 'floor'},
    {'label': 'Cap: 99prc', 'value': 'cap'},
    {"label": "Cap and Floor: 99prc and 1prc", 'value': 'floor_and_cap'},
    {'label': 'Categorize', 'value': 'categorize'},
    {'label': 'Segment', 'value': 'segment'},
    {'label': 'Box-Cox Normalize', 'value': 'boxcox_normalise'},
    {'label': 'Log Transform', 'value': 'log'},
    {'label': 'Log Transform', 'value': 'inverse'},
    {'label': 'Multiply: Col1 x Col2', 'value': 'multiply_by'},
    {'label': 'Divide: Col1 / Col2', 'value': 'divide_by'},
]

PROCESSING_MAPPER = {
    'boxcox_normalise': boxcox_normalise,
    'cap': cap,
    'cap_3std': cap_3std,
    'cat_dummies': cat_dummies,
    'categorize': categorize,
    'floor': floor,
    'floor_and_cap': floor_and_cap,
    'mean_impute': mean_impute,
    'median_impute': median_impute,
    'min_max_scale': min_max_scale,
    'mode_impute': mode_impute,
    'segment': segment,
    'z_normalise': z_normalise,
    "log": log_transform,
    "multiply_by": multiply_by,
    "divide_by": divide_by,
    'inverse': inverse,
}


def parse_contents(contents, filename) -> Union[pd.DataFrame, html.Div]:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            df = None
    except Exception as e:
        return html.Div([
            'There was an error processing this file.'
        ])

    return df


def table_type(df_column):
    # Note - this only works with Pandas >= 1.0.0

    if sys.version_info < (3, 0):  # Pandas 1.0.0 does not support Python 2
        return 'any'

    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
            isinstance(df_column.dtype, pd.BooleanDtype) or
            isinstance(df_column.dtype, pd.CategoricalDtype) or
            isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif (isinstance(df_column.dtype, pd.SparseDtype) or
            isinstance(df_column.dtype, pd.IntervalDtype) or
            isinstance(df_column.dtype, pd.Int8Dtype) or
            isinstance(df_column.dtype, pd.Int16Dtype) or
            isinstance(df_column.dtype, pd.Int32Dtype) or
            isinstance(df_column.dtype, pd.Int64Dtype)):
        return 'numeric'
    else:
        return 'any'


def create_conditional_style(
    df: pd.DataFrame,
    padding: int = 30,
    pixel_for_char: int = 12,
    max_width_pixels: int = 720  # Maximum width as half the page width (assuming 75% scaling)
):
    style = []
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError or TypeError:
                pass
        col_list = df[col].values.tolist()
        col_list = [s if type(s) is str else str(s) for s in col_list]
        col_list.append(col)
        name_length = len(max(col_list, key=len))
        pixel = padding + round(name_length * pixel_for_char)
        pixel = min(pixel, max_width_pixels)  # Cap the width to the maximum value
        pixel = str(pixel) + 'px'
        if pd.api.types.infer_dtype(df[col]) == 'string' or \
                pd.api.types.infer_dtype(df[col]) == 'boolean' and \
                not pd.api.types.is_datetime64_any_dtype(df[col]):
            style.append(
                {
                    'if': {'column_id': col},
                    'minWidth': pixel,
                    'maxWidth': pixel,
                    'width': pixel,
                    'textAlign': 'left',
                }
            )
        else:
            style.append({'if': {'column_id': col}, 'minWidth': pixel})
    return style


app = DashProxy(
    prevent_initial_callbacks=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewpoint", "content": "width=device-width, initial-scale=1"}],
    transforms=[
        MultiplexerTransform(),
    ]
)


app.layout = html.Div([

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    html.Div(id='output-data-upload'),

    dcc.Store(id='stored-data'),
    dcc.Store(id='stored-data-history'),

    dcc.Store(id='stored-column-selector'),
    dcc.Store(id='stored-column-selector-2'),

], style={'margin': '20px'})


@app.callback(
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_data(content, name):
    if content is not None:
        df = parse_contents(content, name)
        if df is not None:
            return df.to_dict('records')
    return dash.no_update


@app.callback(
    Output('output-data-upload', 'children'),
    [
        Input('stored-data', 'data'),
    ], [
        State('upload-data', 'filename'),
        State('stored-column-selector', 'data'),
        State('stored-column-selector-2', 'data'),
    ]
)
def generate_layout(data, file_name, stored_column, stored_column_2):
    if data is not None:
        df = pd.DataFrame(data)
        preview_limit = 1000

        column_styles = []
        colors = {
            'int64': 'lightblue',
            'float64': 'lightgreen',
            'object': 'lightgray',
            'datetime64': 'lightyellow'
        }
        for col_name, dtype in df.dtypes.items():
            color = colors.get(str(dtype), 'white')
            column_styles.append({'if': {'column_id': col_name}, 'backgroundColor': color})

        children = [
            dbc.Card([
                dbc.CardHeader("Data Preview"),
                dbc.CardBody([
                    html.Div([f'Snapshot of {preview_limit} rows from {file_name}:']),
                    dash_table.DataTable(
                        id='table_data',
                        data=df.head(preview_limit).to_dict('records'),
                        columns=[
                            {'name': i, 'id': i, 'type': table_type(df[i])} for i in df.columns
                        ],
                        fixed_rows={'headers': True},
                        style_table={'overflow': 'scroll', 'height': 550},
                        style_cell={
                            'font_size': '12px', 'whiteSpace': 'normal',
                            'height': 'auto'
                        },
                        style_cell_conditional=create_conditional_style(df),
                        style_header={'backgroundColor': '#305D91', 'padding': '10px', 'color': '#FFFFFF'},
                        style_data_conditional=column_styles,
                        editable=True,  # allow editing of data inside all cells
                        filter_action="native",  # allow filtering of data by user ('native') or not ('none')
                        sort_action="native",  # enables data to be sorted per-column by user or not ('none')
                        sort_mode="single",  # sort across 'multi' or 'single' columns
                        column_selectable="multi",  # allow users to select 'multi' or 'single' columns
                        row_selectable="multi",  # allow users to select 'multi' or 'single' rows
                        row_deletable=True,  # choose if user can delete a row (True) or not (False)
                        selected_columns=[],  # ids of columns that user selects
                        selected_rows=[],  # indices of rows that user selects
                        page_action="native",
                    ),
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Div('Color legend:', style={'font-weight': 'bold'}),
                                html.Div('Integer columns',
                                         style={'backgroundColor': colors['int64'], 'display': 'inline-block',
                                                'margin': '5px'}),
                                html.Div('Float columns',
                                         style={'backgroundColor': colors['float64'], 'display': 'inline-block',
                                                'margin': '5px'}),
                                html.Div('String columns',
                                         style={'backgroundColor': colors['object'], 'display': 'inline-block',
                                                'margin': '5px'}),
                                html.Div('Date columns',
                                         style={'backgroundColor': colors['datetime64'], 'display': 'inline-block',
                                                'margin': '5px'})
                            ]), width=11, md=11
                        ),
                        dbc.Col(
                            html.Div([
                                dbc.Button("i", id="filter-help-button", color="info"),
                                dbc.Tooltip(
                                    "When filtering numeric columns please use math operators: i.e. =, <, and >.",
                                    target="filter-help-button",
                                ),
                            ]), width=1, md=1, className='text-right'
                        ),
                    ], justify='between'),
                ])
            ], className='mb-3'),  # add some margin at the bottom

            dbc.Card([
                dbc.CardHeader('Column Selector'),
                dbc.CardBody([
                    dcc.Dropdown(
                        id='drop-column-selector',
                        options=[{'label': i, 'value': i} for i in df.columns],
                        multi=True,
                        value=df.columns.tolist(),  # all columns selected by default
                        placeholder="Select columns to keep...",
                    ),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.Button('Drop Unselected Columns', id='drop-columns-button', n_clicks=0)),
                        dbc.Col(html.Button('Undo', id='undo-drop-button', n_clicks=0))
                    ]),
                ]),
            ], className='mb-3'),

            dbc.Card([
                dbc.CardHeader("Uni-Variate Analysis"),
                dbc.CardBody([
                    html.Div([f'Select a column to analyze:']),
                    dcc.Dropdown(
                        id='column-selector',
                        options=[{'label': i, 'value': i} for i in df.columns],
                        value=stored_column if stored_column is not None else None,
                        placeholder="..."
                    ),
                    html.Br(),
                    html.Div(id='column-statistics'),
                    html.Br(),
                    html.Div(id='column-plot'),
                ])
            ], className='mb-3'),

            dbc.Card([
                dbc.CardHeader("Multi-Variate/Timeseries Analysis"),
                dbc.CardBody([
                    html.Div([f'Select 2nd column for multi-variate analysis:']),
                    dcc.Dropdown(
                        id='column-selector-2',
                        options=[{'label': i, 'value': i} for i in df.columns],
                        value=stored_column_2 if stored_column_2 is not None else None,
                        placeholder="..."
                    ),
                    html.Br(),
                    html.Div(id='column-plot-2'),
                ])
            ], className='mb-3'),

            dbc.Card([
                dbc.CardHeader("Processing Actions"),
                dbc.CardBody([
                    dcc.Store(id='stored-data-history', data=[]),
                    html.Div([f'Select a processing method:']),
                    dcc.Dropdown(id='processing-selector',
                                 options=PROCESSING_OPTIONS,
                                 placeholder="..."),
                    html.Br(),
                    dbc.Row([
                        dbc.Col(html.Button('Apply', id='apply-button', n_clicks=0)),
                        dbc.Col(html.Button('Undo', id='undo-processing-button', n_clicks=0))
                    ]),
                    html.Br(),
                ])
            ]),

            dbc.Card([
                dbc.CardHeader(""),
                dbc.CardBody([
                    dbc.Row(
                        dbc.Col(
                            html.A(
                                id='download-link', download='data.csv', href='', target='_blank',
                                children=[dbc.Button('Download Data', id='download-button', color="danger")]
                            ),
                            width={'size': 6, 'offset': 1}
                        ),
                        justify='center'
                    )
                ])
            ]),
        ]
    else:
        children = [
            dbc.Alert('The uploaded file is not a CSV or Excel file.', color='danger')
        ]
    return children


@app.callback(
    [
        Output('column-statistics', 'children'),
        Output('column-plot', 'children'),
        Output('stored-column-selector', 'data'),
    ], [
        Input('column-selector', 'value')
    ], [
        State('stored-column-selector', 'data'),
        State('stored-data', 'data'),
    ]
)
def update_column_analysis(selected_column, stored_column, data):

    # Ensure that selected and stored columns are in df
    if data is not None:
        df = pd.DataFrame(data)
        if selected_column not in df.columns:
            selected_column = None
        if stored_column not in df.columns:
            stored_column = None

    # Either use selected or stored cols
    if selected_column is None and stored_column is not None:
        selected_column = stored_column
    elif selected_column is not None:
        dcc.Store(id='stored-column-selector', data=selected_column)

    if (selected_column is not None) and (data is not None):
        # df = pd.DataFrame(data)

        # Check if the selected column is numeric or non-numeric
        num_unique = df[selected_column].nunique()
        num_missing = df[selected_column].isna().sum()
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            # Compute statistics for numeric data
            min_value = df[selected_column].min()
            max_value = df[selected_column].max()
            mean_value = df[selected_column].mean()
            std_dev_value = df[selected_column].std()
            kurtosis_value = df[selected_column].kurtosis()
            stats_df = pd.DataFrame(
                {
                    'Metric': ['Min', 'Max', 'Mean', 'Standard Deviation', 'Kurtosis', 'Missing Values'],
                    'Value': [min_value, max_value, mean_value, std_dev_value, kurtosis_value, num_missing]
                }
            )
        else:
            # Compute statistics for non-numeric data
            most_common = df[selected_column].mode().iloc[0] if df[selected_column].mode().size else 'N/A'
            stats_df = pd.DataFrame(
                {
                    'Metric': ['Unique Values', 'Most Common Value', 'Missing Values'],
                    'Value': [num_unique, most_common, num_missing]
                }
            )

        stats_table = dash_table.DataTable(
            data=stats_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in stats_df.columns],
            fixed_rows={'headers': True},
            style_table={'height': '150px', 'overflowY': 'auto'}
        )

        # Generate plot
        if df[selected_column].dtype == 'object' and num_unique <= 30:
            fig = px.bar(df[selected_column].value_counts())
        else:
            fig = px.histogram(df[selected_column])

        plot = dcc.Graph(figure=fig)

        return stats_table, plot, selected_column

    return None, None, None


@app.callback(
    [
        Output('column-plot-2', 'children'),
        Output('stored-column-selector-2', 'data'),
    ], [
        Input('column-selector', 'value'),
        Input('column-selector-2', 'value'),
    ], [
        State('stored-column-selector', 'data'),
        State('stored-column-selector-2', 'data'),
        State('stored-data', 'data'),
    ]
)
def update_column_analysis_2(
    selected_column, selected_column_2,
    stored_column, stored_column_2,
    data,
):
    # Ensure that selected and stored columns are in df
    df: pd.DataFrame
    if data is not None:
        df = pd.DataFrame(data)
        if selected_column not in df.columns:
            selected_column = None
        if stored_column not in df.columns:
            stored_column = None
        if selected_column_2 not in df.columns:
            selected_column_2 = None
        if stored_column_2 not in df.columns:
            stored_column_2 = None

    # Either use selected or stored cols
    if selected_column is None and stored_column is not None:
        selected_column = stored_column

    if selected_column_2 is None and stored_column_2 is not None:
        selected_column_2 = stored_column_2
    elif selected_column_2 is not None:
        dcc.Store(id='stored-column-selector-2', data=selected_column_2)

    if (selected_column is not None) and (selected_column_2 is not None) and (data is not None):
        # df = pd.DataFrame(data)

        # Generate scatter plot
        fig = px.scatter(df, x=selected_column_2, y=selected_column)

        plot = dcc.Graph(figure=fig)

        return plot, selected_column_2

    return None, None


@app.callback(
    [
        Output('stored-data', 'data'),
        Output('stored-data-history', 'data'),
    ], [
        Input('drop-columns-button', 'n_clicks'),
        Input('undo-drop-button', 'n_clicks'),
    ], [
        State('drop-column-selector', 'value'),
        State('stored-data', 'data'),
        State('stored-data-history', 'data'),
    ]
)
def drop_columns(
        apply_clicks, undo_clicks,
        selected_columns, data, data_history
):

    ctx = dash.callback_context
    if ctx.triggered:

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'drop-columns-button':
            df = pd.DataFrame(data)
            data_history.append(data)  # store the current state in the history

            df = df[selected_columns]  # keep only selected columns
            return df.to_dict('records'), data_history

        elif button_id == 'undo-drop-button' and data_history:
            data = data_history.pop()  # remove the last state from the history and use it as the current state
            return data, data_history

    raise dash.exceptions.PreventUpdate  # prevent update if button hasn't been clicked


@app.callback(
    [
        Output('stored-data', 'data'),
        Output('stored-data-history', 'data'),
    ], [
        Input('apply-button', 'n_clicks'),
        Input('undo-processing-button', 'n_clicks')],
    [
        State('column-selector', 'value'),
        State('processing-selector', 'value'),
        State('stored-data', 'data'),
        State('stored-data-history', 'data'),
        State('stored-column-selector', 'data'),
    ]
)
def update_or_undo_column_processing(
        apply_clicks,
        undo_clicks,
        selected_column,
        processing,
        data,
        data_history,
        stored_column,
):

    ctx = dash.callback_context
    if ctx.triggered:

        if selected_column is None and stored_column is not None:
            selected_column = stored_column

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'apply-button' and selected_column is not None and processing is not None and data is not None:
            df = pd.DataFrame(data)
            data_history.append(data)  # store the current state in the history

            processor = PROCESSING_MAPPER[processing]
            outputs = processor(df, col=selected_column)
            if type(outputs) == pd.DataFrame:
                df = outputs
            elif type(outputs) == dict:
                df = outputs.pop('df')
            else:
                raise ValueError(
                    f'update_or_undo_column_processing got '
                    f'an unexpected output data type ({type(outputs)}) after '
                    f'applying the processing function: {PROCESSING_MAPPER[processing]}. '
                    f'Note that the function supports dict or pd.Dataframe outputs.'
                )

            return df.to_dict('records'), data_history

        elif button_id == 'undo-processing-button' and data_history:
            data = data_history.pop()  # remove the last state from the history and use it as the current state
            return data, data_history

    return data, data_history  # no change if no button was pressed


@app.callback(
    Output('download-link', 'href'),
    Input('download-button', 'n_clicks'),
    State('stored-data', 'data'),
)
def generate_download_url(n_clicks, stored_data):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate

    # Convert the stored data back to a DataFrame
    df = pd.DataFrame(stored_data)

    # Convert the DataFrame to a CSV and encode it
    csv_string = df.to_csv(index=False, encoding='utf-8')
    csv_data_uri = 'data:text/csv;charset=utf-8,' + urllib.parse.quote(csv_string)

    return csv_data_uri


if __name__ == '__main__':
    app.run_server(debug=True)
