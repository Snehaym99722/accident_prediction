# SafeRoute: Accident Severity Prediction

A Python Streamlit dashboard that trains a Random Forest classifier from the PowerPoint case summaries.

## Dashboard

- Overview of case totals, severity classes, and test accuracy.
- Severity prediction for a selected recorded factor, category, and period.
- Model insights with confusion matrix, classification report, and feature importance.
- Browse and download the case data used by the model.
- Adjust the test-data portion in the sidebar.

The model uses the included PowerPoint summary counts. Choose a recorded factor such as collision type, time, season, violation, vehicle, or junction, then select a category and period to predict severity.

## Run

    python -m pip install -r requirements.txt
    python -m streamlit run app.py --server.port 8507

