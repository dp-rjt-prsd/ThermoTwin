# Build every result the dashboard and README use, in order (about 10 minutes).
# Run from the repository root with the virtual environment active:
#   .\scripts\build_all_results.ps1
# Output folders must not exist yet; delete old ones in results/ and data/processed/ first.
$ErrorActionPreference = "Stop"

function Step($label, [string[]]$arguments) {
    Write-Host "==> $label"
    & python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $label" }
}

Step "1/9 Simulate 60 normal training runs" @("scripts/generate_dataset.py", "--seed", "1729", "--trajectory-count", "60", "--output-directory", "data/processed/id")
Step "2/9 Simulate 12 harsher test runs" @("scripts/generate_dataset.py", "--ood", "--seed", "1730", "--trajectory-count", "12", "--output-directory", "data/processed/ood")
Step "3/9 Train both digital twins" @("scripts/compare_models.py", "--id-dataset-directory", "data/processed/id", "--ood-dataset-directory", "data/processed/ood", "--output-directory", "results/phase12-final-comparison")
Step "4/9 Check forecasts on normal data" @("scripts/evaluate_forecasts.py", "--comparison-directory", "results/phase12-final-comparison", "--dataset-directory", "data/processed/id", "--output-directory", "results/forecast-id")
Step "5/9 Check forecasts on harsher data" @("scripts/evaluate_forecasts.py", "--comparison-directory", "results/phase12-final-comparison", "--dataset-directory", "data/processed/ood", "--output-directory", "results/forecast-ood")
Step "6/9 Compare four controllers over six 10-minute scenarios" @("scripts/benchmark.py", "--seed", "1729", "--comparison-directory", "results/phase12-final-comparison", "--output-directory", "results/benchmark")
Step "7/9 Resistance-fault detection case" @("scripts/run_fault_detection.py", "--comparison-directory", "results/phase12-final-comparison", "--model-variant", "physics_informed", "--seed", "1729", "--output-directory", "results/fault")
Step "8/9 Final evaluation summary and charts" @("scripts/final_evaluation.py", "--benchmark-directory", "results/benchmark", "--comparison-directory", "results/phase12-final-comparison", "--id-forecast-directory", "results/forecast-id", "--ood-forecast-directory", "results/forecast-ood", "--fault-directory", "results/fault/fault-detection", "--output-directory", "results/phase12-final-evaluation")
Step "9/9 Dashboard controller run" @("scripts/run_predictive.py", "--comparison-directory", "results/phase12-final-comparison", "--seed", "1729", "--output-directory", "results/predictive")
Write-Host "All results built. Start the dashboard with:"
Write-Host "  python scripts/run_dashboard.py --predictive-directory results/predictive --fault-directory results/fault --evaluation-directory results/phase12-final-evaluation"
