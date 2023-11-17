\name{LR_GradientConjugate_gpumatrix}
\title{Logistic Regression with Conjugate Gradient method}

\description{
The developed function performs the logistic regression using the Conjugate Gradient method. This method has shown to be very effective for logistic regression of big models  [1]. The code is general enough to accommodate standard R matrices, sparse matrices from the 'Matrix' package and, more interestingly, gpu.matrix-class objects from the GPUmatrix package.

}

\alias{LR_GradientConjugate_gpumatrix}
\usage{
LR_GradientConjugate_gpumatrix(X, y, beta = NULL,
                               lambda = 0, iterations = 100,
                               tol = 1e-08)
}

\arguments{
  \item{X}{the design matrix. Could be either a object of class \code{\linkS4class{gpu.matrix}}, \code{\linkS4class{matrix}}, or  \code{\linkS4class{Matrix}}.}
  \item{y}{vector of observations.}
  \item{beta}{initial solution.}
  \item{lambda}{numeric, the penalty per parameter to be used.}
  \item{iterations}{maximum number of iterations.}
  \item{tol}{tolerance to be used for the estimation.}
}


\value{
The function returns a vector containing the values of the coefficients. This returned vector will be a 'matrix', 'Matrix' or 'gpu.matrix-class' object depending on the class of the object \code{X}.
}

\author{
Angel Rubio and Cesar Lobato.
}

\references{
[1] Minka TP (2003). “A comparison of numerical optimizers for logistic regression.” URL: https://tminka.github.io/papers/logreg/minka-logreg.pdf.
}

\examples{
a <- 5
}

