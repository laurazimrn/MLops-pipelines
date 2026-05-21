"""Simple drift detection stub using Evidently.
This file provides a minimal example that can be extended.
"""

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except Exception:
    Report = None
    DataDriftPreset = None


def detect_drift(reference_df, current_df, out_html='drift_report.html'):
    if Report is None:
        print('Evidently not installed. Install `evidently` to enable drift detection.')
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(out_html)
    print('Drift report saved to', out_html)
