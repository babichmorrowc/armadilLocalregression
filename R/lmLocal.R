#' Local least squares regression with Gaussian kernel
#'
#' @description This function is used to fit a local linear regression model with Gaussian kernel weights.
#' For a given \eqn{\mathbf{x}_0}, this function estimates \eqn{\hat{\beta}(\mathbf{x}_0)}, such that
#' \deqn{\hat{\beta}(\mathbf{x}_0) = \arg\min_{\beta} \sum_{i=1}^n \kappa_{\mathbf{H}}(\mathbf{x}_0 - \mathbf{x}_i) (y_i - \tilde{\mathbf{x}}_i^\top \beta)^2}
#' where \eqn{\kappa_H} is a density kernel with positive definite bandwidth matrix \eqn{\mathbf{H}}.
#'
#' @param y Output variable
#' @param x0 Matrix of the predictor variables at a single prediction location
#' @param X0 Matrix of all the covariates at the prediction location
#' @param x Matrix of the predictor variables
#' @param X Matrix of all the covariates
#' @param H Positive definite bandwidth matrix
#'
#' @return The estimated regression coefficients \eqn{\hat \beta(\mathbf{x}_0)} at the prediction locations
#' @export
#'
#' @examples
lmLocal <- function(y, x0, X0, x, X, H){
  w <- mvtnorm::dmvnorm(x, x0, H)
  fit <- lm(y ~ -1 + X, weights = w)
  return( t(X0) %*% coef(fit) )
}

#' Local least squares regression with Gaussian kernel
#'
#' @description This function fits a local linear regression model with Gaussian kernel weights.
#' The regression coefficients are a function of the covariates \eqn{\mathbf{x}}, i.e. \eqn{\hat{\beta} = \hat{\beta}(\mathbf{x})}.
#' For a given \eqn{\mathbf{x}_0}, \deqn{\hat{\beta}(\mathbf{x}_0) = \arg\min_{\beta} \sum_{i=1}^n \kappa_{\mathbf{H}}(\mathbf{x}_0 - \mathbf{x}_i) (y_i - \tilde{\mathbf{x}}_i^\top \beta)^2}
#' where \eqn{\kappa_H} is a density kernel with positive definite bandwidth matrix \eqn{\mathbf{H}}.
#'
#' @param y Output variable
#' @param x0 Matrix of the predictor variables at the prediction locations
#' @param X0 Matrix of all the covariates at the prediction locations
#' @param x Matrix of the predictor variables
#' @param X Matrix of all the covariates
#' @param H Positive definite bandwidth matrix
#'
#' @return The estimated regression coefficients \eqn{\hat \beta(\mathbf{x}_0)} at the prediction locations
#' @export
#'
#' @examples
predLocal <- function(y, x0, X0, x, X, H) {
  nsub <- nrow(x0)
  preds <- sapply(1:nsub, function(ii){
    lmLocal(y = y, x0 = x0[ii, ], X0 = X0[ii, ], x = x, X = X, H = H)
  })
  return(preds)
}
