import warnings
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd
from evidently import Report
from evidently.core.datasets import DataDefinition, Dataset, Regression
from evidently.metrics import RMSE, ValueDrift
from evidently.presets import RegressionPreset
from evidently.tests import lt
from evidently.ui.workspace import Workspace
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

# Suppress this known Evidently 0.7.2/pandas deprecation throughout the script.
warnings.filterwarnings(
    "ignore",
    message=r"DataFrameGroupBy\.apply operated on the grouping columns.*",
    category=FutureWarning,
    module=r"evidently\.legacy\.metrics\.regression_performance\.utils",
)

# Evidently's internal regression plots can contain bins with only one row.
# R-squared is undefined for those bins; the overall MAE/RMSE remain valid.
warnings.filterwarnings(
    "ignore",
    message=r"R\^2 score is not well-defined with less than two samples\.",
    category=UndefinedMetricWarning,
    module=r"sklearn\.metrics\._regression",
)

TARGET = "cnt"
FEATURES = [
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]


def download_data():
    data_url = (
        "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
    )
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    hour_path = data_dir / "hour.csv"
    day_path = data_dir / "day.csv"
    zip_path = data_dir / "bike_sharing.zip"

    if not hour_path.is_file() or not day_path.is_file():
        if not zip_path.is_file():
            urlretrieve(data_url, zip_path)

        with ZipFile(zip_path) as archive:
            archive.extractall(data_dir)

    df_hour = pd.read_csv(hour_path)
    df_day = pd.read_csv(day_path)
    return df_hour, df_day


def prepare_data(hourly_data):
    data = hourly_data.copy()
    data["dteday"] = pd.to_datetime(data["dteday"])
    dates = data["dteday"]
    reference = data.loc[(dates.dt.year == 2011) & (dates.dt.month == 1)].copy()
    production = data.loc[(dates.dt.year == 2011) & (dates.dt.month == 2)].copy()
    return reference, production


def train_and_evaluate(reference):
    reference = reference.sort_values(["dteday", "hr"])
    X = reference[FEATURES]
    y = reference[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=False
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Separate model inputs from the enriched tables used for monitoring.
    train_data = X_train.copy()
    train_data[TARGET] = y_train
    train_data["prediction"] = train_predictions
    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data["prediction"] = test_predictions

    print(f"Training rows: {len(train_data)}, test rows: {len(test_data)}")
    print(f"January test MAE: {mean_absolute_error(y_test, test_predictions):.2f}")
    print(f"January test RMSE: {root_mean_squared_error(y_test, test_predictions):.2f}")
    return model, train_data, test_data


def validate_model(train_data, test_data):
    definition = DataDefinition(
        regression=[Regression(target="cnt", prediction="prediction")]
    )
    test_dataset = Dataset.from_pandas(test_data, data_definition=definition)

    report = Report(
        metrics=[RMSE(tests=[lt(50)])],
        include_tests=True,
    )
    snapshot = report.run(current_data=test_dataset)
    return snapshot


def initialize_workspace():
    project_root = Path(__file__).resolve().parent.parent
    workspace = Workspace.create(str(project_root / "bike_sharing_monitoring"))

    project_name = "Bike Rental Monitoring"
    projects = workspace.search_project(project_name)
    project = projects[0] if projects else workspace.create_project(project_name)

    return workspace, project


def prepare_monitoring_weeks(model, production):
    boundaries = [
        ("W1", "2011-02-01", "2011-02-08"),
        ("W2", "2011-02-08", "2011-02-15"),
        ("W3", "2011-02-15", "2011-02-22"),
    ]
    weeks = {}

    for name, start, end in boundaries:
        mask = (production["dteday"] >= pd.Timestamp(start)) & (
            production["dteday"] < pd.Timestamp(end)
        )
        week = production.loc[mask].sort_values(["dteday", "hr"]).copy()
        week["prediction"] = model.predict(week[FEATURES])
        weeks[name] = week

    return weeks


def monitor_weeks(reference, weeks, workspace, project):
    definition = DataDefinition(
        regression=[Regression(target="cnt", prediction="prediction")]
    )
    reference_dataset = Dataset.from_pandas(
        reference[FEATURES + [TARGET, "prediction"]],
        data_definition=definition,
    )

    snapshots = {}

    for name, week in weeks.items():
        current_dataset = Dataset.from_pandas(
            week[FEATURES + [TARGET, "prediction"]],
            data_definition=definition,
        )

        report = Report(
            metrics=[RegressionPreset()],
            tags=[name],
        )
        snapshot = report.run(
            reference_data=reference_dataset,
            current_data=current_dataset,
            timestamp=week["dteday"].max().to_pydatetime(),
        )
        workspace.add_run(project.id, snapshot)
        snapshots[name] = snapshot

    return snapshots


def compare_weekly_performance(weeks):
    rows = []

    for name, week in weeks.items():
        rows.append(
            {
                "week": name,
                "MAE": mean_absolute_error(week[TARGET], week["prediction"]),
                "RMSE": root_mean_squared_error(week[TARGET], week["prediction"]),
            }
        )

    performance = pd.DataFrame(rows).set_index("week")
    print(performance.round(2))

    worst_week = performance["RMSE"].idxmax()
    print(f"Most degraded week by RMSE: {worst_week}")
    return worst_week


def diagnose_drift(reference, week, week_name, workspace, project):
    weather_columns = ["temp", "atemp", "hum", "windspeed"]
    columns = [TARGET] + weather_columns

    definition = DataDefinition(numerical_columns=columns)

    reference_dataset = Dataset.from_pandas(
        reference[columns],
        data_definition=definition,
    )
    current_dataset = Dataset.from_pandas(
        week[columns],
        data_definition=definition,
    )

    report = Report(
        metrics=[ValueDrift(column=column) for column in columns],
        tags=[week_name, "diagnosis"],
    )
    snapshot = report.run(
        reference_data=reference_dataset,
        current_data=current_dataset,
        timestamp=week["dteday"].max().to_pydatetime(),
    )

    workspace.add_run(project.id, snapshot)

    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "diagnostic_report.html"
    snapshot.save_html(str(output_path))
    print(f"Diagnostic report saved to: {output_path}")

    return snapshot


if __name__ == "__main__":
    df_hour, _ = download_data()
    reference, production = prepare_data(df_hour)
    print(f"January rows: {len(reference)}, February rows: {len(production)}")
    model, train_data, test_data = train_and_evaluate(reference)
    validation_snapshot = validate_model(train_data, test_data)
    workspace, project = initialize_workspace()
    workspace.add_run(project.id, validation_snapshot)

    weeks = prepare_monitoring_weeks(model, production)
    # for name, week in weeks.items():
    #     print(f"{name}: {len(week)} rows")
    weekly_snapshots = monitor_weeks(test_data, weeks, workspace, project)
    worst_week = compare_weekly_performance(weeks)
    diagnostic_snapshot = diagnose_drift(
        reference,
        weeks[worst_week],
        worst_week,
        workspace,
        project,
    )
