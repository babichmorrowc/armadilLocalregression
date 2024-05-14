// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

//' Linear regression using QR decomposition
//'
//' @description Fit a linear regression model with `RcppArmadillo` using QR decomposition.
//'
//' @param X Model matrix of dimension n x p
//' @param y Response vector of length n
//'
//' @return Coefficients of the linear model
//' @export
//'
//' @examples
//' data(mtcars)
//' y <- mtcars$mpg
//' X <- with(mtcars, cbind(1, wt, wt^2, hp, hp^2))
//' beta_hat <- armadillo_lm(X, y)
//' beta_hat
// [[Rcpp::export(name = "armadillo_lm")]]
arma::vec armadillo_lm(arma::mat& X, arma::vec& y) {
  arma::mat Q;
  arma::mat R;

  qr_econ(Q, R, X); // QR decomposition of X
  arma::vec beta = solve(R, (trans(Q) * y)); // Solve the system R * beta = Q^T * y for beta
  return beta;
}

//' Evaluate multivariate Gaussian density
//'
//' @description Analogous to the `dmvnorm` function in the `mvtnorm` package.
//'
//' @param X matrix of quantiles
//' @param mu vector of means
//' @param L lower triangular matrix of the Cholesky decomposition of the covariance matrix
//'
//' @return vector of density values
arma::vec dmvnInt(arma::mat& X, const rowvec& mu, arma::mat& L)
{
  unsigned int d = X.n_cols;
  unsigned int m = X.n_rows;

  arma::vec D = L.diag();
  // Define vector that will contain the density values
  arma::vec out(m);
  arma::vec z(d);

  double acc;
  unsigned int icol, irow, ii;
  for(icol = 0; icol < m; icol++) // Loop over the x values
  {
    for(irow = 0; irow < d; irow++) // Loop over the dimensions
    {
      acc = 0.0;
      for(ii = 0; ii < irow; ii++) acc += z.at(ii) * L.at(irow, ii);
      z.at(irow) = ( X.at(icol, irow) - mu.at(irow) - acc ) / D.at(irow);
    }
    out.at(icol) = sum(square(z));
  }

  // Compute the density
  out = exp( - 0.5 * out - ( (d / 2.0) * log(2.0 * M_PI) + sum(log(D)) ) );

  return out;
}

//' Local regression model
//'
//' @description This function is used to fit a local regression model with Gaussian kernel weights using `RcppArmadillo`.
//' For a given \eqn{\mathbf{x}_0}, this function estimates \eqn{\hat{\beta}(\mathbf{x}_0)}, such that
//' \deqn{\hat{\beta}(\mathbf{x}_0) = \arg\min_{\beta} \sum_{i=1}^n \kappa_{\mathbf{H}}(\mathbf{x}_0 - \mathbf{x}_i) (y_i - \tilde{\mathbf{x}}_i^\top \beta)^2}
//' where \eqn{\kappa_H} is a density kernel with positive definite bandwidth matrix \eqn{\mathbf{H}}.
//'
//' @param y Output variable
//' @param x0 Matrix of the predictor variables at a single prediction location
//' @param X0 Matrix of all the covariates at the prediction location
//' @param x Matrix of the predictor variables
//' @param X Matrix of all the covariates
//' @param H Positive definite bandwidth matrix
//'
//' @return The estimated regression coefficients \eqn{\hat \beta(\mathbf{x}_0)} at the prediction locations
//' @export
//'
//' @examples
//' data(mtcars)
//' y <- mtcars$mpg
//' x <- as.matrix(mtcars[, c("wt", "hp")])
//' X <- with(mtcars, cbind(1, wt, wt^2, hp, hp^2))
//' H <- diag(2) # use identity bandwidth matrix for example
//' # Predict at three locations
//' preds <- armadillo_lm_local(y= y, x0 = x[1:3,], X0 = X[1:3,], x = x, X = X, H = H)
// [[Rcpp::export(name = "armadillo_lm_local")]]
 arma::vec armadillo_lm_local( arma::vec& y, arma::mat& x0, arma::mat& X0, arma::mat& x, arma::mat& X, arma::mat& H) {

   // Get L for use in dmvnInt
   arma::mat L = chol(H, "lower");
   // Get the number of observations
   int nrow = x0.n_rows;
   // Vector of fits
   arma::vec fitted(nrow);
   // Vector of weights
   arma::vec weights(nrow);

   for (int i = 0; i < nrow; i++) {
     // Get the weights
     weights = dmvnInt(x, x0.row(i), L);
     // Get the fitted values
     arma::mat X_weights = X.each_col() % sqrt(weights);
     arma::vec y_weights = y % sqrt(weights);
     arma::vec fit = armadillo_lm(X_weights, y_weights);
     arma::mat X0_fit = X0.row(i) * fit;
     fitted(i) = X0_fit(0);
   }

   return fitted;
 }
