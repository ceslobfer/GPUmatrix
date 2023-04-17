\name{matrix_decomposition}
\title{Decomposition of a matrix with GPU}

\description{

These functions compute different decompositions of a matrix using the GPU.

\code{'eigen'} mimics the base function \code{'eigen'} that computes the eigenvalues and eigenvectors of a numeric (double, integer, logical) or complex matrix.

\code{'svd'} mimics the base function \code{'svd'} that computes the singular-value decomposition of a rectangular matrix.

\code{'qr'} mimics the base function \code{'qr'} that computes the QR decomposition of a matrix.


\code{'chol'} mimics the base function \code{'chol'} that computes Compute the Cholesky factorization of a real symmetric positive-definite square matrix.

}

\alias{eigen}
\alias{eigen-methods}
\alias{eigen,gpu.matrix.tensorflow-method}
\alias{eigen,gpu.matrix.torch-method}

\alias{svd}
\alias{svd-methods}
\alias{svd,gpu.matrix.tensorflow-method}
\alias{svd,gpu.matrix.torch-method}

\alias{qr}
\alias{qr-methods}
\alias{qr,gpu.matrix.tensorflow-method}
\alias{qr,gpu.matrix.torch-method}

\alias{chol}
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

\item{x}{a numeric or complex gpu.matrix whose spectral decomposition or whose
SVD decomposition or whose QR decomposition or whose Cholesky factorization is
to be computed. For the Cholesky factorization, x must be real symmetric
positive-definite square gpu.matrix.}

}


\seealso{
For more information see:

\code{\link[base]{eigen}},
 \code{\link[base]{svd}},
 \code{\link[base]{chol}},
 \code{\link[base]{qr}},
 \code{\link{solve}}

}

\value{
The output of these functions correspond to their equivalent base functions:

\code{eigen} mimics the base function \code{eigen} that computes the eigenvalues and eigenvectors of a numeric (double, integer, logical) or complex matrix. It returns a list with the following items:
\item{values}{a vector with the \code{P} eigenvalues of \code{x}}
\item{vectors}{the eigenvectors of \code{x}}

\code{svd} mimics the base function \code{svd} that computes the singular-value decomposition of a rectangular matrix. It returns a list with the following items:
\item{d}{a vector containing the singular values of \code{x}}
\item{u}{a matrix whose columns contain the left singular vectors of \code{x}}
\item{v}{a matrix whose columns contain the right singular vectors of \code{x}}

\code{qr} mimics the base function \code{qr} that computes the QR decomposition of a matrix. It returns a list with the following items:
\item{qr}{a matrix with the same dimensions as x. The upper triangle contains the \bold{R} of the decomposition and the lower triangle contains information on the \bold{Q} of the decomposition}
\item{qraux}{a vector of length \code{ncol(x)} which contains additional information on \bold{Q}.}
\item{rank}{the rank of \code{x} as computed by the decomposition(*): always full rank in the LAPACK case.}
\item{pivot}{information on the pivoting strategy used during the decomposition.}


\code{chol} mimics the base function \code{chol} that computes Compute the Cholesky factorization of a real symmetric positive-definite square matrix. It returns a GPUmatrix object with The upper triangular factor of the Cholesky decomposition, i.e., the matrix R such that R'R = x.
}

\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(9),3,3)
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

