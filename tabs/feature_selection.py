import pandas as pd
import tkinter as tk
from tkinter import ttk
from pandas import DataFrame
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class FeatureSelection:

    def __init__(self):
        pass

    def calculate_feature_importance(self, train_data, target_column, features):
        import xgboost as xgb
        import pandas as pd
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        X_train = train_data[features]
        y_train = train_data[target_column]

        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)

        feature_importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({'feature': features, 'importance': feature_importances})

        # Create a new figure and a subplot.
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot()

        # Plot the feature importance.
        feature_importance_df.sort_values(
            'importance', ascending=False
        ).plot(kind='bar', x='feature', y='importance', ax=ax)

        # Create the canvas.
        canvas = FigureCanvasTkAgg(fig, master=self)  # A tk.DrawingArea.

        # Pack the canvas into the Frame.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        return feature_importance_df

    def calculate_correlation_matrix(self, data):
        # TODO: implement correlation matrix calculation
        return pd.DataFrame()


class FeatureSelectionTab(ttk.Frame):

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.feature_selection = FeatureSelection()
        self.create_widgets()

    def create_widgets(self):
        # Use padding to add some space around widgets
        padding = {'padx': 5, 'pady': 5}

        # Use frames to group related widgets together
        features_frame = ttk.Frame(self)
        features_frame.pack(**padding)

        # Style labels for a more professional look
        style = ttk.Style()
        style.configure('TLabel', font=('Arial', 11))

        # Field for specifying feature columns
        self.feature_label = ttk.Label(features_frame, text="Specify initial feature set")
        self.feature_label.pack(side=tk.LEFT)
        self.feature_var = tk.StringVar(self)
        self.feature_menu = ttk.OptionMenu(features_frame, self.feature_var, '')
        self.feature_menu.pack(side=tk.LEFT, **padding)

        # Field for specifying target column
        self.target_label = ttk.Label(features_frame, text="Specify target column")
        self.target_label.pack(side=tk.LEFT)
        self.target_var = tk.StringVar(self)
        self.target_menu = ttk.OptionMenu(features_frame, self.target_var, '')
        self.target_menu.pack(side=tk.LEFT, **padding)

        # Button for performing feature selection
        fs_button_frame = ttk.Frame(self)
        fs_button_frame.pack(**padding)
        self.fs_button = ttk.Button(
            fs_button_frame, text="Perform feature selection",
            command=self.perform_feature_selection
        )
        self.fs_button.pack(side=tk.LEFT, **padding)

        # Treeview for feature importances
        self.feature_importances_tree = ttk.Treeview(self)
        self.feature_importances_tree.pack()

        # Figure for correlation matrix
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_feature_selection_widgets(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

        # Update feature menu
        self.feature_menu['menu'].delete(0, 'end')
        self.target_menu['menu'].delete(0, 'end')

        # Use train_data instead of data to update the menus
        for column in train_data.columns:
            self.feature_menu['menu'].add_command(
                label=column, command=tk._setit(self.feature_var, column)
            )
            self.target_menu['menu'].add_command(
                label=column, command=tk._setit(self.target_var, column)
            )

    def perform_feature_selection(self):
        if self.train_data is None or self.test_data is None:
            raise ValueError("Train and test data must be set")

        target_column = self.target_var.get()  # get the selected target column
        features = self.feature_var.get()  # get the selected features

        feature_importances = self.feature_selection.calculate_feature_importance(
            self.train_data, target_column, features
        )

        # TODO: Display the feature_importances in an interactive bar chart

