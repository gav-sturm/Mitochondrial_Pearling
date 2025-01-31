import pandas as pd
import napari
from napari.layers import Labels, Image
from PyQt5.QtWidgets import (
    QPushButton,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QComboBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QHeaderView,
)
from PyQt5.QtCore import Qt, QTimer
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.environ['NAPARI_LOG_LEVEL'] = 'WARNING'

class AnnotationWidget(QWidget):
    def __init__(self, categories, label_column, annotated_metrics, live_annotation_list, parent=None):
        super().__init__(parent)
        self.categories = categories
        self.label_column = label_column
        self.annotated_metrics = annotated_metrics
        self.live_annotation_list = live_annotation_list
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Instruction label
        instruction_label = QLabel(
            "Select a label and choose its category from the dropdown."
        )
        layout.addWidget(instruction_label)

        # Dropdown for categories
        self.category_dropdown = QComboBox()
        self.category_dropdown.addItems(self.categories)
        layout.addWidget(self.category_dropdown)

        # Display selected label
        self.selected_label_label = QLabel("Selected Label: None")
        layout.addWidget(self.selected_label_label)

        # Annotation table
        self.annotation_table = QTableWidget()
        self.annotation_table.setColumnCount(2)
        self.annotation_table.setHorizontalHeaderLabels(["Label ID", "Pearl Type"])
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        self.annotation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.annotation_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.annotation_table.setSelectionMode(QTableWidget.SingleSelection)
        self.annotation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.annotation_table)

        # Delete button
        self.delete_button = QPushButton("Delete Annotation")
        self.delete_button.setFixedHeight(40)
        self.delete_button.clicked.connect(self.delete_annotation)
        layout.addWidget(self.delete_button)

        self.setLayout(layout)

    def assign_category(self, selected_label, selected_category):
        if selected_label == 0 or selected_label is None:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select a valid label to annotate.",
            )
            return
        self.annotated_metrics.loc[
            self.annotated_metrics[self.label_column] == selected_label, "pearl_type"
        ] = selected_category
        print(f"Label {selected_label} marked as '{selected_category}'.")
        self.selected_label_label.setText(f"Selected Label: {selected_label}")
        self.populate_table()
        self.live_annotation_list.update_table()

    def populate_table(self):
        """
        Populates the annotation table with current annotations.
        """
        self.annotation_table.setRowCount(0)  # Clear existing rows
        for idx, row in self.annotated_metrics.iterrows():
            label_id = row[self.label_column]
            pearl_type = row["pearl_type"]
            self.annotation_table.insertRow(
                self.annotation_table.rowCount()
            )
            label_item = QTableWidgetItem(str(label_id))
            pearl_item = QTableWidgetItem(
                pearl_type if pd.notna(pearl_type) else ""
            )
            self.annotation_table.setItem(
                self.annotation_table.rowCount() - 1, 0, label_item
            )
            self.annotation_table.setItem(
                self.annotation_table.rowCount() - 1, 1, pearl_item
            )

    def delete_annotation(self):
        selected_rows = self.annotation_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(
                self,
                "No Selection",
                "Please select an annotation to delete.",
            )
            return
        for selected_row in selected_rows:
            row_idx = selected_row.row()
            label_id = int(
                self.annotation_table.item(row_idx, 0).text()
            )
            # Set pearl_type to 'non_pearl'
            self.annotated_metrics.loc[
                self.annotated_metrics[self.label_column] == label_id, "pearl_type"
            ] = "non_pearl"
            print(f"Annotation for label {label_id} deleted (set to 'non_pearl').")
        self.populate_table()
        self.live_annotation_list.update_table()

class LiveAnnotationList(QWidget):
    def __init__(self, label_column, annotated_metrics, annotation_widget, labels_layer, parent=None):
        super().__init__(parent)
        self.label_column = label_column
        self.annotated_metrics = annotated_metrics
        self.annotation_widget = annotation_widget
        self.labels_layer = labels_layer
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Label
        label = QLabel("Current Annotations:")
        layout.addWidget(label)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Label ID", "Pearl Type"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.cellClicked.connect(self.on_cell_clicked)
        layout.addWidget(self.table)

        self.setLayout(layout)

    def on_cell_clicked(self, row, column):
        label_id = int(self.table.item(row, 0).text())
        # Select the corresponding label in napari
        self.labels_layer.selected_label = label_id
        # Update the Annotation Controls' selected label label
        self.annotation_widget.selected_label_label.setText(f"Selected Label: {label_id}")

    def update_table(self):
        """
        Updates the live annotation table with current annotations.
        """
        self.table.setRowCount(0)  # Clear existing rows
        for idx, row in self.annotated_metrics.iterrows():
            label_id = row[self.label_column]
            pearl_type = row["pearl_type"]
            self.table.insertRow(self.table.rowCount())
            label_item = QTableWidgetItem(str(label_id))
            pearl_item = QTableWidgetItem(
                pearl_type if pd.notna(pearl_type) else ""
            )
            self.table.setItem(
                self.table.rowCount() - 1, 0, label_item
            )
            self.table.setItem(
                self.table.rowCount() - 1, 1, pearl_item
            )

class PearlClassifier:
    def __init__(
        self,
        csv_path,
        label_column="label_id",
        test_size=0.2,
        random_state=42,
    ):
        """
        Initializes the PearlClassifier with the given CSV file.

        Parameters:
        - csv_path (str): Path to the CSV file containing the data.
        - label_column (str): The name of the column containing the label IDs.
        - test_size (float): Proportion of the dataset to include in the validation split.
        - random_state (int): Random seed for reproducibility.
        """
        self.csv_path = csv_path
        self.label_column = label_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state)
        self.is_trained = False
        self.model_path = "pearl_classifier_model.joblib"
        self.feature_names = None  # To store feature names after loading data
        self.pearl_label_metrics = None  # To store label metrics

    def load_data(self):
        """
        Loads data from the CSV file and separates features and labels.

        Returns:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Labels.
        """
        try:
            data = pd.read_csv(self.csv_path)
            if self.label_column not in data.columns:
                raise ValueError(
                    f"Label column '{self.label_column}' not found in the CSV."
                )

            # Initialize 'pearl_type' as 'non_pearl' if not already set
            if "pearl_type" not in data.columns:
                data["pearl_type"] = "non_pearl"
            else:
                data["pearl_type"].fillna("non_pearl", inplace=True)

            X = data.drop(columns=[self.label_column, "pearl_type"])
            y = data["pearl_type"]
            self.feature_names = X.columns.tolist()  # Store feature names for future use
            return X, y
        except FileNotFoundError:
            print(f"File {self.csv_path} not found.")
            raise
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise

    def train(self):
        """
        Trains the RandomForestClassifier using the loaded data.
        Splits the data into training and validation sets, fits the model, and evaluates accuracy.
        """
        X, y = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Validation Accuracy: {accuracy:.2f}")

    def save_model(self, path=None):
        """
        Saves the trained model to disk.

        Parameters:
        - path (str): Path to save the model. If None, uses default path.
        """
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")

        save_path = path if path else self.model_path
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, save_path)
        print(f"Model saved to {save_path}")

    def load_model_from_file(self, path=None):
        """
        Loads a trained model from disk.

        Parameters:
        - path (str): Path to the saved model. If None, uses default path.
        """
        load_path = path if path else self.model_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}.")

        loaded = joblib.load(load_path)
        self.model = loaded["model"]
        self.feature_names = loaded.get("feature_names", None)
        self.is_trained = True
        print(f"Model loaded from {load_path}")

    def predict(self, input_data):
        """
        Makes predictions on new data using the trained model.

        Parameters:
        - input_data (pd.DataFrame or str): New data for prediction. If a string is provided, it is treated as a path to a CSV file.

        Returns:
        - predictions (np.ndarray): Predicted labels.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained. Please train or load a model before prediction."
            )

        if isinstance(input_data, str):
            # Assume input_data is a path to a CSV file
            input_data = pd.read_csv(input_data)

        # Ensure that input_data has the same feature columns
        if self.feature_names:
            missing_features = set(self.feature_names) - set(input_data.columns)
            if missing_features:
                raise ValueError(f"Input data is missing features: {missing_features}")
            input_data = input_data[self.feature_names]

        predictions = self.model.predict(input_data)
        return predictions

    def predict_single(self, sample):
        """
        Makes a prediction on a single sample using the trained model.

        Parameters:
        - sample (dict or pd.Series or pd.DataFrame): A dictionary, pandas Series, or single-row DataFrame representing the sample's morphological features.

        Returns:
        - prediction (str): The predicted label for the sample.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model is not trained. Please train or load a model before prediction."
            )

        if isinstance(sample, dict):
            sample_df = pd.DataFrame([sample])
        elif isinstance(sample, pd.Series):
            sample_df = sample.to_frame().T
        elif isinstance(sample, pd.DataFrame):
            if sample.shape[0] != 1:
                raise ValueError("DataFrame should contain exactly one sample (one row).")
            sample_df = sample
        else:
            raise TypeError(
                "Sample must be a dict, pandas Series, or a single-row DataFrame."
            )

        # Ensure that sample_df has the same feature columns
        if self.feature_names:
            missing_features = set(self.feature_names) - set(sample_df.columns)
            if missing_features:
                raise ValueError(f"Input sample is missing features: {missing_features}")
            sample_df = sample_df[self.feature_names]

        prediction = self.model.predict(sample_df)[0]
        return prediction

    def annotate_labels(
            self, pearl_label_metrics, image, labels, label_column="label_id", pearl_order=False
    ):
        """
        Allows the user to interactively annotate labels as 'pearl', 'split_pearl', 'merged_pearl', or 'non_pearl' using napari.

        Parameters:
        - pearl_label_metrics (pd.DataFrame): DataFrame containing label metrics and a 'pearl_type' column.
        - image (np.ndarray): Image array to display.
        - labels (np.ndarray): Labeled mask array corresponding to the image.
        - label_column (str): Column name in pearl_label_metrics that matches label IDs in labels.

        Returns:
        - annotated_metrics (pd.DataFrame): Updated DataFrame with 'pearl_type' annotations.
        """
        from napari.layers import Labels
        from napari.qt import thread_worker

        # Validate inputs
        if label_column not in pearl_label_metrics.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in pearl_label_metrics."
            )

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != 0]  # Assuming 0 is background

        # Ensure all labels are present in the DataFrame
        missing_labels = set(unique_labels) - set(pearl_label_metrics[label_column])
        if missing_labels:
            raise ValueError(
                f"The following labels are missing in pearl_label_metrics: {missing_labels}"
            )

        # Initialize napari viewer
        viewer = napari.Viewer(title="Pearl Annotation")

        # Add image layer
        viewer.add_image(image, name="Image", opacity=0.8)

        # Prepare annotation DataFrame
        annotated_metrics = pearl_label_metrics.copy()

        # Add centroids as points and label them with label_id
        centroids = np.array(
            annotated_metrics['location_pixel'].tolist())  # Assuming 'location_pixel' is a list of coordinates
        label_ids = annotated_metrics[label_column].tolist()

        # Add points at the centroid locations
        viewer.add_points(centroids, size=1, name="Centroids", face_color='red')

        # Add text annotations for label IDs at the centroids
        viewer.add_points(centroids, size=0, name="Label IDs", visible=False, text={
            'string': [str(label_id) for label_id in label_ids],  # Convert label IDs to strings
            'size': 12,
            'color': 'white',
            'anchor': 'center'
        })

        # Add labels layer with selection enabled
        labels_layer = viewer.add_labels(
            labels, name="Labels", opacity=0.5, blending="translucent"
        )
        viewer.layers.selection.active.mode = 'pick' # for older version of napari
        # viewer.window.qt_viewer.canvas.set_active_tool("pick") # for newer version of napari > 0.5.2



        # Define available categories
        categories = ["non_pearl", "pearl", "split_pearl", "merged_pearl"]
        if pearl_order:
            categories = ["pearl", "non_pearl", "split_pearl", "merged_pearl"]
        # Instantiate LiveAnnotationList first
        live_annotation_list = LiveAnnotationList(
            label_column=label_column,
            annotated_metrics=annotated_metrics,
            annotation_widget=None,  # Placeholder
            labels_layer=labels_layer
        )

        # Instantiate AnnotationWidget
        annotation_widget = AnnotationWidget(
            categories=categories,
            label_column=label_column,
            annotated_metrics=annotated_metrics,
            live_annotation_list=live_annotation_list
        )

        # Link live_annotation_list back to annotation_widget
        live_annotation_list.annotation_widget = annotation_widget

        # Add dock widgets to napari
        viewer.window.add_dock_widget(
            annotation_widget, area="right", name="Annotation Controls"
        )
        viewer.window.add_dock_widget(
            live_annotation_list, area="left", name="Live Annotations"
        )

        # Populate tables initially
        annotation_widget.populate_table()
        live_annotation_list.update_table()

        # Initialize last_selected_label to track changes
        self.last_selected_label = None

        # Create a QTimer to periodically check the selected label
        timer = QTimer()
        timer.setInterval(250)  # Check every 500 milliseconds

        def check_selected_label():
            selected_label = labels_layer.selected_label
            if selected_label != 0 and selected_label is not None:
                if selected_label != self.last_selected_label:
                    self.last_selected_label = selected_label
                    current_category = annotation_widget.category_dropdown.currentText()
                    annotation_widget.assign_category(selected_label, current_category)

        timer.timeout.connect(check_selected_label)
        timer.start()

        # Set 'Pick mode' as the default tool
        try:
            viewer.window.qt_viewer.canvas.set_active_tool("pick")
            print("Set 'Pick Mode' as the active tool.")
        except AttributeError:
            print("Could not set 'Pick Mode' as the active tool. Please select it manually.")

        # Instructions for the user
        print("Instructions for Annotation:")
        print("1. **Select Tool**: 'Pick Mode' is already selected.")
        print(
            "2. **Select a Label**: Click on a labeled region in the image or select a row in the 'Live Annotations' table.")
        print("   - Selecting a label will automatically assign the current category to it.")
        print(
            "3. **Choose Category**: From the dropdown menu in the 'Annotation Controls' on the right, select the desired category.")
        print("   - Changing the category will affect future label selections.")
        print("4. **Live Updates**: The 'Live Annotations' table on the left will update automatically.")
        print(
            "5. **Delete Annotation**: To remove an annotation, select it in the 'Annotation Controls' table and click 'Delete Annotation'.")
        print("6. **Finalize**: Close the napari window to finish and see the annotations.")

        # Start napari
        napari.run()

        is_pearl_category = ['pearl', 'split_pearl', 'merged_pearl']
        annotated_metrics['is_pearl'] = [True if pearl_type in is_pearl_category else False for pearl_type in
                                         annotated_metrics['pearl_type']]

        # After napari closes, return the annotations
        return annotated_metrics

