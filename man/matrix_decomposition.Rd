\name{matrix_decomposition}
\title{Decomposition of a Matrix with GPU}

\description{

These function compute different decomposition of a matrix using the GPU.

\code{eigen} mimics the base function \code{eigen} that computes the eigenvalues and eigenvectors of a numeric (double, integer, logical) or complex matrix.

\code{svd} mimics the base function \code{svd} that computes the singular-value decomposition of a rectangular matrix.

\code{qr} mimics the base function \code{qr} that computes the QR decomposition of a matrix.


\code{chol} mimics the base function \code{chol} that computes Compute the Cholesky factorization of a real symmetric positive-definite square matrix.

}

\alias{eigen-methods}
\alias{eigen,gpu.matrix.tensorflow-method}
\alias{eigen,gpu.matrix.torch-method}

\alias{svd-methods}
\alias{svd,gpu.matrix.tensorflow-method}
\alias{svd,gpu.matrix.torch-method}

\alias{qr-methods}
\alias{qr,gpu.matrix.tensorflow-method}
\alias{qr,gpu.matrix.torch-method}

\alias{chol-methods}
\alias{chol,gpu.matrix.tensorflow-method}
\alias{chol,gpu.matrix.torch-method}

\usage{

\S4method{eigen}{gpu.matrix.tensorflow}(x)
\S4method{eigen}{gpu.matrix.torch}(x)

\S4method{svd}{gpu.matrix.tensorflow}(x)
\S4method{svd}{gpu.matrix.torch}(x)

\S4method{chol}{gpu.matrix.tensorflow}(x)
\S4method{chol}{gpu.matrix.torch}(x)

\S4method{qr}{gpu.matrix.tensorflow}(x)
\S4method{qr}{gpu.matrix.torch}(x)

}


\arguments{

\item{x}{a numeric or complex gpu.matrix whose spectral decomposition or whose SVD decomposition or whose QR decomposition or whose Cholesky factorization is to be computed. For the Cholesky factorization, x must be real symmetric positive-definite square gpu.matrix.}

}


\seealso{
For more information see:

\code{\link[base]{eigen}},
 \code{\link[base]{svd}},
 \code{\link[base]{chol}},
 \code{\link[base]{qr}},
 \code{\link{solve}}

}


\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(9),3,3,type="tensorflow")
ein <- eigen(a) #eigenvalues and eigenvectors
svd_return <- svd(a) #svd of gpu.matrix a
qr_return <- qr(a) #qr decompostion

ata <- tcrossprod(a)
#ata is a real symmetric positive-definite square matrix.
chol(ata) #cholesky decomposition.

#for an example of how to use the cholesky decomposition see
#the corresponding help for the function chol_solve.

}
}
}

