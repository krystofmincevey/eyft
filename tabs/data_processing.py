import pandas as pd
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')

from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.stats import kurtosis, skew
from typing import Tuple, Callable

from src.eyft.pipelines.data_processing.processor import (
    boxcox_normalise, cap, cap_3std, cat_dummies,
    categorize, floor, floor_and_cap, mean_impute,
    median_impute, min_max_scale, mode_impute,
    segment, z_normalise,
)
from src.eyft.pipelines.feature_engineering.transform import (
    log_transform, inverse
)
from src.eyft.utils.generic import BoundedList


class DataProcessing:

    def __init__(self):
        self.processing_mapper = PROCESSING_MAPPER = {
            'Box-Cox Normalize': boxcox_normalise,
            'Floor: 99prc': cap,
            'Removal Outliers: 3 STD': cap_3std,
            'Create Dummy': cat_dummies,
            'Categorize': categorize,
            'Floor: 1prc': floor,
            'Cap and Floor: 99prc and 1prc': floor_and_cap,
            'Mean Impute': mean_impute,
            'Median Impute': median_impute,
            'Min-Max Scale': min_max_scale,
            'Mode Impute': mode_impute,
            'Segment': segment,
            'Z-Normalize': z_normalise,
            "Log Transform": log_transform,
            'inverse': inverse,
        }

    def process(
            self, proc_key: str, process_col: str,
            df_train: pd.DataFrame, df_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if proc_key not in self.processing_mapper:
            raise ValueError(f"Invalid processing function: {proc_key}.")

        proc = self.processing_mapper[proc_key]
        outputs = proc(df_train, process_col)

        if type(outputs) == dict:  # for functions from data_processing
            df_train = outputs.pop('df')  # need pop to avoid passing to proc
            if df_test is not None:
                df_test = proc(df_test, **outputs)['df']
        else:  # for functions from feature_engineering
            df_train = outputs
            if df_test is not None:
                df_test = proc(df_test, col=process_col)

        return df_train, df_test


class DataProcessingTab(ttk.Frame):

    def __init__(self, parent, set_split_data: Callable):
        ttk.Frame.__init__(self, parent)

        self.data_processing = DataProcessing()
        self.set_split_data: Callable = set_split_data # method to pass split data back to app

        self.problem_type: str or None = None
        self.data: pd.DataFrame or None = None
        self.train_data: pd.DataFrame or None = None
        self.test_data: pd.DataFrame or None = None
        self.train_data_history: BoundedList(pd.DataFrame) = BoundedList(3)
        self.test_data_history: BoundedList(pd.DataFrame) = BoundedList(3)
        self.ax_list = []

        self.create_widgets()

    def create_widgets(self):

        def configure_scroll_region(event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        padding = {'padx': 5, 'pady': 5}
        scroll_canvas = self._create_scroll_canvas()
        main_frame = self._create_main_frame(scroll_canvas)

        main_frame.bind("<Configure>", configure_scroll_region)

        self._create_timeseries_widgets(main_frame, padding)
        self._create_split_widgets(main_frame, padding)
        self._create_column_widgets(main_frame, padding)
        self._create_secondary_column_selection_widgets(main_frame, padding)
        self._create_tertiary_column_selection_widgets(main_frame, padding)
        self._create_plot_widgets(main_frame, padding)
        self._create_stats_table(main_frame)
        self._create_fig_and_canvas(main_frame)
        self._create_processing_options_widgets(main_frame, padding)
        self._create_buttons_widgets(main_frame, padding)

    def _create_scroll_canvas(self):
        scroll_canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=scroll_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return scroll_canvas

    def _create_main_frame(self, scroll_canvas):
        main_frame = ttk.Frame(scroll_canvas)
        scroll_canvas.create_window((0, 0), window=main_frame, anchor='nw')
        return main_frame

    def _create_timeseries_widgets(self, main_frame, padding):
        timeseries_frame = ttk.Frame(main_frame)
        timeseries_frame.pack(fill=tk.BOTH, expand=True, **padding)
        timeseries_frame.grid_columnconfigure(0, weight=1)
        timeseries_frame.grid_columnconfigure(1, weight=1)
        self.order_label = ttk.Label(timeseries_frame, text="Set Column for Timeseries Ordering:")
        self.order_label.grid(row=0, column=0, sticky='ew', **padding)
        self.order_combobox = ttk.Combobox(timeseries_frame)
        self.order_combobox.grid(row=0, column=1, sticky='ew', **padding)
        self.order_label.grid_remove()
        self.order_combobox.grid_remove()

    def _create_split_widgets(self, main_frame, padding):
        split_frame = ttk.Frame(main_frame)
        split_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.split_label = ttk.Label(split_frame, text="Set a Test Percentage:")
        self.split_label.pack(side=tk.LEFT)
        self.split_var = tk.DoubleVar()
        self.split_var.trace('w', self.update_split_label)
        self.split_slider = ttk.Scale(split_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.split_var)
        self.split_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, **padding)
        self.split_value_label = ttk.Label(split_frame, text='')
        self.split_value_label.pack(side=tk.LEFT)

    def _create_column_widgets(self, main_frame, padding):
        column_frame = ttk.Frame(main_frame)
        column_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.column_label = ttk.Label(column_frame, text="Select a Column to Analyse:")
        self.column_label.pack(side=tk.LEFT)
        self.column_combobox = ttk.Combobox(column_frame)
        self.column_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, **padding)

    def _create_secondary_column_selection_widgets(self, main_frame, padding):
        secondary_column_frame = ttk.Frame(main_frame)
        secondary_column_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.secondary_column_label = ttk.Label(secondary_column_frame, text="Select a Scatter By Column:")
        self.secondary_column_label.pack(side=tk.LEFT)

        self.secondary_column_combobox = ttk.Combobox(secondary_column_frame)
        self.secondary_column_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, **padding)

    def _create_tertiary_column_selection_widgets(self, main_frame, padding):
        tertiary_column_frame = ttk.Frame(main_frame)
        tertiary_column_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.tertiary_column_label = ttk.Label(tertiary_column_frame, text="Select a Group By Column:")
        self.tertiary_column_label.pack(side=tk.LEFT)

        self.tertiary_column_combobox = ttk.Combobox(tertiary_column_frame)
        self.tertiary_column_combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, **padding)

    def _create_plot_widgets(self, main_frame, padding):
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.plot_button = ttk.Button(
            plot_frame, text="Analyse selected column", command=self.analyze
        )
        self.plot_button.pack(side=tk.LEFT, **padding)

    def _create_stats_table(self, main_frame):
        # Treeview for train statistics table
        self.tree = ttk.Treeview(main_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def _create_fig_and_canvas(self, main_frame):
        # Create the initial figure
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)  # A tk.DrawingArea.

        # Add a scrollbar
        scrollbar = tk.Scrollbar(main_frame, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.get_tk_widget().config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.canvas.get_tk_widget().yview)

        self.canvas.draw()

        # Add matplotlib navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
        toolbar.update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _create_processing_options_widgets(self, main_frame, padding):
        # Processing options menu in its own frame
        processing_options_frame = ttk.Frame(main_frame)
        processing_options_frame.pack(fill=tk.BOTH, expand=True, **padding)

        self.processing_options_title = ttk.Label(
            processing_options_frame, text="Select a Processing Function:"
        )
        self.processing_options_title.pack(fill=tk.X, expand=True)

        supported_processing_functions = list(
            self.data_processing.processing_mapper.keys()
        )  # Assuming processing_mapper is a dictionary
        self.processing_options_menu = ttk.Combobox(
            processing_options_frame, values=supported_processing_functions
        )
        self.processing_options_menu.pack(fill=tk.X, expand=True)

    def _create_buttons_widgets(self, main_frame, padding):
        # Apply and undo buttons in their own frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.BOTH, expand=True, **padding)
        self.apply_button = tk.Button(buttons_frame, text="Apply", command=self.apply_processing)
        self.apply_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.undo_button = tk.Button(buttons_frame, text="Undo", command=self.undo_processing)
        self.undo_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def update_widgets(self, data, problem_type):
        if data is None or problem_type is None:
            raise ValueError("Data or problem type cannot be None")

        self.data = data
        self.problem_type = problem_type

        # Get column names
        column_names = list(self.data.columns)

        # Update order combobox for timeseries problem
        if self.problem_type == 'timeseries':
            self.order_label.grid()
            self.order_combobox['values'] = column_names
            self.order_combobox.grid()
        else:
            self.order_label.grid_remove()
            self.order_combobox.grid_remove()

        # Update column combobox
        self.column_combobox['values'] = column_names
        self.secondary_column_combobox['values'] = column_names
        self.tertiary_column_combobox['values'] = column_names

    def update_split_label(self, *args):
        self.split_value_label['text'] = f"{self.split_var.get():.2f}%"  # Show the value with 2 decimal places

    def split_data(self):
        if self.data is None or self.problem_type is None:
            raise ValueError("Data or problem type cannot be None")

        if self.problem_type == 'timeseries':

            if not self.order_combobox.get():
                raise ValueError("Order var must be set")

            order_column = self.order_combobox.get()
            self.data = self.data.sort_values(by=order_column)

        train_pct = 1 - (self.split_slider.get() / 100)  # imputed as prc [0-100]
        split_index = int(len(self.data) * train_pct)

        self.train_data = self.data[:split_index]
        self.test_data = self.data[split_index:]

        self.set_split_data(self.train_data, self.test_data)

    def join_data(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("Train or Test data cannot be None.")

        # Reset the original data
        self.data = pd.concat([self.train_data, self.test_data])

    def analyze(self):
        primary_column = self.column_combobox.get()
        secondary_column = self.secondary_column_combobox.get()
        tertiary_column = self.tertiary_column_combobox.get()

        if not primary_column:
            raise ValueError("Primary column to analyse not specified")

        # Get train and test data
        self.split_data()

        # Generate subplots
        total_plots = 2  # Minimum of 2 plots for the primary column (train and test)
        if secondary_column:
            total_plots += 2  # Additional 2 plots if secondary column selected
        if tertiary_column:
            total_plots += 2  # Additional 1 plot if tertiary column selected

        # Create all subplots upfront
        if total_plots != len(self.ax_list):
            # Clear the existing figure and create a new one
            self.fig.clf()
            self.ax_list = [
                self.fig.add_subplot(
                    int(total_plots / 2), 2, i + 1
                ) for i in range(total_plots)
            ]
            # Adjust subplot parameters to give specified padding
            self.fig.subplots_adjust(wspace=0.3, hspace=0.3)

        self._plot(secondary_column, tertiary_column)

    def _plot(
        self, secondary_column=None, tertiary_column=None
    ):

        # Generate the statistics table
        self.generate_stats_table()

        # Generate plots for primary column
        self.plot_primary_column()

        if secondary_column:
            # Generate plots for secondary column
            self.plot_secondary_column()

        if tertiary_column:
            # Generate plot for tertiary column
            self.plot_tertiary_column()

        # Redraw the canvas after adding the interactivity
        self.canvas.draw_idle()

    def plot_primary_column(self):
        column = self.column_combobox.get()
        total_plots = len(self.ax_list)

        for i, (data, title, ax) in enumerate(
                [
                    (self.train_data, 'Train Data', self.ax_list[0]),
                    (self.test_data, 'Test Data', self.ax_list[1]),
                ]
        ):
            ax.clear()
            ax.set_title(title, fontsize=14)
            # If column is categorical and has less than or equal to 30 unique values, use bar plot
            if data[column].dtype == 'object' and data[column].nunique() <= 30:
                value_counts = data[column].value_counts().sort_values()
                ax.bar(value_counts.index, value_counts.values, color='skyblue')
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')  # rotate x-tick labels
            else:  # otherwise, use histogram
                ax.hist(data[column].dropna(), bins=30, color='skyblue', alpha=0.7)

            ax.set_ylabel('Count', fontsize=12)  # Set y-label to "Count"
            if total_plots <= 2:
                ax.set_xlabel(column, fontsize=12)  # Set x-label to selected column's name

    def plot_secondary_column(self):
        primary_column = self.column_combobox.get()
        secondary_column = self.secondary_column_combobox.get()
        total_plots = len(self.ax_list)

        for i, (data, title, ax) in enumerate(
                [
                    (self.train_data, 'Train Data', self.ax_list[2]),
                    (self.test_data, 'Test Data', self.ax_list[3]),
                ]
        ):
            ax.clear()
            ax.scatter(data[primary_column], data[secondary_column], color='skyblue', alpha=0.7)
            ax.set_ylabel(secondary_column, fontsize=12)
            if total_plots <= 4:
                ax.set_xlabel(primary_column, fontsize=12)

    def plot_tertiary_column(self):
        primary_column = self.column_combobox.get()
        tertiary_column = self.tertiary_column_combobox.get()

        for i, (data, title, ax) in enumerate(
                [
                    (self.train_data, 'Train Data', self.ax_list[-2]),
                    (self.test_data, 'Test Data', self.ax_list[-1]),
                ]
        ):
            ax.clear()
            boxplot_data = [group[primary_column].values for name, group in data.groupby(tertiary_column)]
            ax.boxplot(boxplot_data, vert=False)
            ax.set_ylabel(tertiary_column, fontsize=12)
            ax.set_xlabel(primary_column, fontsize=12)

    def generate_stats_table(self):
        # Define the column names and headings
        column = self.column_combobox.get()
        column_width = 120
        headings = ["Metric", "Train Value", "Test Value"]

        # Initialize tree
        self.tree["columns"] = headings
        self.tree.column("#0", width=0, stretch=False)
        for col in headings:
            self.tree.column(col, width=column_width, minwidth=column_width, anchor="w")
            self.tree.heading(col, text=col, anchor="w")

        # Compute statistics for train and test data
        data_sets = [(self.train_data, 'Train Value'), (self.test_data, 'Test Value')]
        all_stats = {}
        for data, _ in data_sets:
            num_unique = data[column].nunique()
            num_missing = data[column].isna().sum()
            if pd.api.types.is_numeric_dtype(data[column]):
                min_value = data[column].min()
                q1 = data[column].quantile(0.25)
                median_value = data[column].median()
                q3 = data[column].quantile(0.75)
                max_value = data[column].max()
                mean_value = data[column].mean()
                std_dev_value = data[column].std()
                skewness_value = skew(data[column].dropna())
                kurtosis_value = kurtosis(data[column].dropna())
                stats = [('Min', min_value), ('25th Percentile (Q1)', q1), ('Median', median_value),
                         ('75th Percentile (Q3)', q3), ('Max', max_value), ('Mean', mean_value),
                         ('Standard Deviation', std_dev_value), ('Skewness', skewness_value),
                         ('Kurtosis', kurtosis_value), ('Unique Values', num_unique),
                         ('Missing Values', num_missing)]
            else:
                most_common = data[column].mode()[0]
                stats = [
                    ('Unique Values', num_unique), ('Most Common Value', most_common),
                    ('Missing Values', num_missing)
                ]
            all_stats[_] = stats

        # Clear the old table
        for i in self.tree.get_children():
            self.tree.delete(i)

        # Add the data
        for train_stat, test_stat in zip(all_stats['Train Value'], all_stats['Test Value']):
            self.tree.insert("", "end", values=(train_stat[0], train_stat[1], test_stat[1]))

    def apply_processing(self):

        selected_column = self.column_combobox.get()
        selected_processing = self.processing_options_menu.get()

        if not selected_column or not selected_processing:
            raise ValueError(
                f'Processing method: {selected_processing} or '
                f'column: {selected_column} to analyse not set.'
            )

        # Split the data - done anew to allow for processing without analysis
        self.split_data()

        # Store the current state in the history
        self.train_data_history.append(self.train_data.copy())
        self.test_data_history.append(self.test_data.copy())

        # Apply processing
        self.train_data, self.test_data = self.data_processing.process(
            proc_key=selected_processing, process_col=selected_column,
            df_train=self.train_data.copy(), df_test=self.test_data.copy(),
        )

        # Update the stored data
        self.join_data()

        # Update figure.
        secondary_column = self.secondary_column_combobox.get()
        tertiary_column = self.tertiary_column_combobox.get()
        self._plot(secondary_column, tertiary_column)

    def undo_processing(self):
        try:
            # Remove the last state from the history and use it as the current state
            self.train_data = self.train_data_history.pop()
            self.test_data = self.test_data_history.pop()

            # Reset the data.
            self.join_data()

            # Reset figure.
            secondary_column = self.secondary_column_combobox.get()
            tertiary_column = self.tertiary_column_combobox.get()
            self._plot(secondary_column, tertiary_column)
        except IndexError:
            raise IndexError('Nothing to undo as processing history is empty.')
