# Burgers figure provenance

The main-text figure uses the saved Burgers–MLP seed-2 checkpoint and the stored time nearest the fixed choice t = 0.5: time index 100, t = 0.5025125628140703, on the original 200 × 400 time–space grid. No model or Jacobian is evaluated.

Source directory: `C:\dev\pinn-uq-jmlr-ready\results\burgers_mlp_seed2`.

- `burgers_mlp_seed2_fields.npz`: x, time, predicted solution and reference solution; absolute error is their pointwise absolute difference.
- `burgers_mlp_seed2_fullgrid_sigma_raw.npz`: full-network predictive standard deviation and saved sharp/bulk masks under the training-weighted residual metric. The standard deviation in the separate historical fields file is not used.
- `C:\dev\pinn-uq-jmlr-ready\calibration_figures\fullgrid_sigma_metric_contract.json` records the source hashes and covariance scope.

The source grids agree; arrays follow time-major meshgrid order. `burgers_slice.csv` retains all 400 x values with absolute error, sigma and the original full-grid region masks. Eight slice points are sharp and 392 are bulk. No new slice-specific percentile is computed. Values are positive, so plotting uses no clipping, added floor or rescaling. The inviscid breaking time 1/pi provides a time reference for this viscous Burgers example.

Run `python analysis/plot_burgers_slice.py` with NumPy and Matplotlib to recreate the vector PDF and 300 dpi PNG in `figures`. The figure illustrates a single slice, not the full-grid RMS ratios reported in the tables.
