import base64
import io
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px

from typing import Union

from dash import dcc, html, dash_table
from dash_extensions.enrich import (
    DashProxy, MultiplexerTransform,
    Input, Output, State,
)


PROCESSING_OPTIONS = [
    {'label': 'Mean Imputation', 'value': 'mean'},
    {'label': 'Median Imputation', 'value': 'median'},
    {'label': 'Mode Imputation', 'value': 'mode'},
    {'label': 'Min-Max Scaling', 'value': 'minmax'},
    {'label': 'Z-Normalization', 'value': 'znorm'},
    {'label': 'Outlier Capping', 'value': 'outliers'}
]

external_stylesheets = [dbc.themes.BOOTSTRAP]


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
    ],
    [
        State('upload-data', 'filename'),
        State('stored-column-selector', 'data'),
        State('stored-column-selector-2', 'data'),
    ]
)
def generate_layout(data, file_name, stored_column, stored_column_2):
    if data is not None:
        df = pd.DataFrame(data)

        # DEBUG:
        col = stored_column if stored_column is not None else None

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
                    html.Div([f'Viewing: {file_name}']),
                    dash_table.DataTable(
                        data=df.head(1000).to_dict('records'),
                        columns=[{'name': col_name, 'id': col_name} for col_name in df.columns],
                        fixed_rows={'headers': True},
                        style_cell={'width': '150px'},  # adjust as needed
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        style_data_conditional=column_styles,
                        style_header={
                            'textOverflow': 'ellipsis',  # use 'auto' for text wrapping
                        },
                    ),
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
                    ])
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
                    html.Button('Drop Unselected Columns', id='drop-columns-button', n_clicks=0)
                ]),
            ], className='mb-3'),

            dbc.Card([
                dbc.CardHeader("Uni-Variate Analysis"),
                dbc.CardBody([
                    html.Div([f'Select a column to analyze:  debug: {col}']),
                    dcc.Dropdown(
                        id='column-selector',
                        options=[{'label': i, 'value': i} for i in df.columns],
                        # value=stored_column if stored_column is not None else '...',
                        value=col,
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
                        # value=stored_column_2 if stored_column_2 is not None else '...',
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
                        dbc.Col(html.Button('Undo', id='undo-button', n_clicks=0))
                    ]),
                    html.Br(),
                ])
            ])
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
    ],
    [
        Input('column-selector', 'value')
    ],
    [
        State('stored-column-selector', 'data'),
        State('stored-data', 'data'),
    ]
)
def update_column_analysis(selected_column, stored_column, data):

    if selected_column is None and stored_column is not None:
        selected_column = stored_column
    elif selected_column is not None:
        dcc.Store(id='stored-column-selector', data=selected_column)

    if (selected_column is not None) and (data is not None):
        df = pd.DataFrame(data)

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
    Output('column-plot-2', 'children'),
    [
        Input('column-selector', 'value'),
        Input('column-selector-2', 'value'),
        Input('stored-data', 'data'),
        Input('stored-column-selector', 'data'),
        Input('stored-column-selector-2', 'data')
    ]
)
def update_column_analysis_2(selected_column, selected_column_2, data, stored_column, stored_column_2):

    if selected_column is None and stored_column is not None:
        selected_column = stored_column

    if selected_column_2 is None and stored_column_2 is not None:
        selected_column_2 = stored_column_2
    elif selected_column_2 is not None:
        dcc.Store(id='stored-column-selector-2', data=selected_column_2)

    if (selected_column is not None) and (selected_column_2 is not None) and (data is not None):
        df = pd.DataFrame(data)

        # Generate scatter plot
        fig = px.scatter(df, x=selected_column_2, y=selected_column)

        plot = dcc.Graph(figure=fig)

        return plot

    return None


@app.callback(
    [
        Output('stored-data', 'data'),
        Output('stored-data-history', 'data'),
    ], [
        Input('apply-button', 'n_clicks'),
        Input('undo-button', 'n_clicks')],
    [
        State('column-selector', 'value'),
        State('column-selector-2', 'value'),
        State('processing-selector', 'value'),
        State('stored-data', 'data'),
        State('stored-data-history', 'data'),
        State('stored-column-selector', 'data'),
        State('stored-column-selector-2', 'data')
    ]
)
def update_or_undo_column_processing(
        apply_clicks,
        undo_clicks,
        selected_column,
        selected_column_2,
        processing,
        data,
        data_history,
        stored_column,
        stored_column_2
):

    ctx = dash.callback_context
    if ctx.triggered:

        if selected_column is None and stored_column is not None:
            selected_column = stored_column
        elif selected_column is not None:
            dcc.Store(id='stored-column-selector', data=selected_column)

        if selected_column_2 is None and stored_column_2 is not None:
            selected_column_2 = stored_column_2
        elif selected_column_2 is not None:
            dcc.Store(id='stored-column-selector-2', data=selected_column_2)

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'apply-button' and selected_column is not None and processing is not None and data is not None:
            df = pd.DataFrame(data)
            data_history.append(data)  # store the current state in the history

            if processing == 'mean':
                df[selected_column].fillna(df[selected_column].mean(), inplace=True)
            elif processing == 'median':
                df[selected_column].fillna(df[selected_column].median(), inplace=True)
            elif processing == 'mode':
                df[selected_column].fillna(df[selected_column].mode()[0], inplace=True)
            elif processing == 'minmax':
                min_val = df[selected_column].min()
                max_val = df[selected_column].max()
                df[selected_column] = (df[selected_column] - min_val) / (max_val - min_val)
            elif processing == 'znorm':
                mean = df[selected_column].mean()
                std = df[selected_column].std()
                df[selected_column] = (df[selected_column] - mean) / std
            elif processing == 'outlier':
                # Here we cap outliers to the 1st and 99th percentile. You can adjust this as needed.
                lower = df[selected_column].quantile(0.01)
                upper = df[selected_column].quantile(0.99)
                df[selected_column] = np.where(df[selected_column] < lower, lower, df[selected_column])
                df[selected_column] = np.where(df[selected_column] > upper, upper, df[selected_column])

            return df.to_dict('records'), data_history

        elif button_id == 'undo-button' and data_history:
            data = data_history.pop()  # remove the last state from the history and use it as the current state
            return data, data_history

    return data, data_history  # no change if no button was pressed


if __name__ == '__main__':
    app.run_server(debug=True)




#
#
# # @app.callback(
# #     [
# #         Output('output-data-upload', 'children', True),  # The True here indicates multiplexed output
# #         Output('stored-data', 'data', True),  # The True here indicates multiplexed output
# #     ], [Input('drop-columns-button', 'n_clicks')],
# #     [
# #         State('drop-column-selector', 'value'),
# #         State('stored-data', 'data')
# #     ]
# # )
# # def drop_columns(n_clicks, selected_columns, data):
# #     if n_clicks > 0:  # button has been clicked
# #         df = pd.DataFrame(data)
# #         df = df[selected_columns]  # keep only selected columns
# #
# #         # update table view
# #         children = dash_table.DataTable(
# #             data=df.head(1000).to_dict('records'),
# #             columns=[{'name': i, 'id': i} for i in df.columns],
# #             fixed_rows={'headers': True},
# #             style_cell={'width': '150px'},  # adjust as needed
# #             style_table={'height': '300px', 'overflowY': 'auto'},
# #             style_header={'textOverflow': 'ellipsis'},
# #         )
# #
# #         return children, df.to_dict('records')
# #
# #     raise dash.exceptions.PreventUpdate  # prevent update if button hasn't been clicked
#
#
# @app.callback(
#     [
#         Output('stored-data', 'data', True),  # The True here indicates multiplexed output
#         Output('column-selector', 'options'),  # Update column-selector dropdown options
#         Output('drop-column-selector', 'options'),  # Update drop-column-selector dropdown options
#     ],
#     [Input('drop-columns-button', 'n_clicks')],
#     [
#         State('drop-column-selector', 'value'),
#         State('stored-data', 'data')
#     ]
# )
# def drop_columns(n_clicks, selected_columns, data):
#     if n_clicks > 0:  # button has been clicked
#         df = pd.DataFrame(data)
#         df = df[selected_columns]  # keep only selected columns
#
#         # update column-selector and drop-column-selector dropdown options
#         column_options = [{'label': i, 'value': i} for i in df.columns]
#
#         return df.to_dict('records'), column_options, column_options
#
#     raise dash.exceptions.PreventUpdate  # prevent update if button hasn't been clicked
