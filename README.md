# armadilLocalregression

  <!-- badges: start -->
  [![R-CMD-check](https://github.com/babichmorrowc/armadilLocalregression/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/babichmorrowc/armadilLocalregression/actions/workflows/R-CMD-check.yaml)
  <!-- badges: end -->

This package contains functions for performing linear regression using `RcppArmadillo`, as well as functions for local regression (using both `R` and `RcppArmadillo`).

For linear regression functions, see the `armadillo_lm` function. For local regression functions, the `R` functions are `lmLocal` and `predLocal`, and the `RcppArmadillo` function is `armadillo_lm_local` (with an additional function `armadillo_cv_H` for performing cross-validation for the bandwidth matrix).
