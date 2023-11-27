\name{matrix_decomposition}
\title{Decomposition of a matrix with GPU}

\description{

These functions mimic the functions \code{eigen} ,\code{svd},\code{chol} to operate on gpu.matrix-class objects:

\code{'eigen'} mimics the base function \code{'eigen'} that "computes the eigenvalues and eigenvectors of a numeric (double, integer, logical) or complex matrix."

\code{'svd'} mimics the base function \code{'svd'} that "computes the singular-value decomposition of a rectangular matrix."

\code{'chol'} mimics the base function \code{'chol'} that "computes Compute the Cholesky factorization of a real symmetric positive-definite square matrix."

}

\alias{eigen}
\alias{eigen-methods}
\alias{eigen,gpu.matrix.tensorflow-method}
\alias{eigen,gpu.matrix.torch-method}

\alias{svd}
\alias{svd-methods}
\alias{svd,gpu.matrix.tensorflow-method}
\alias{svd,gpu.matrix.torch-method}

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

}


\arguments{
\item{x}{a \code{\linkS4class{gpu.matrix}}. \code{X} must fulfil certain characteristics depending on the function to be called (see details).}
}

\details{

These functions mimic the behaviour of their respective 'base' functions.

In the case of the \code{eigen} function, the input value can be a numeric or complex gpu.matrix class.

For \code{svd} function, the input value could be a numeric or complex gpu.matrix-class object.

For \code{chol} function, the input must be a positive-definite squere matrix.


Internally, these functions call its corresponding function of the tensorflow or torch library depending on the type of input gpu.matrix-class.

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}

\seealso{
For more information see:
\code{\link[base]{eigen}}, \code{\link[base]{svd}}, \code{\link[base]{chol}}, \code{\link[torch]{linalg_eig}}, \code{\link[torch]{torch_svd}}, and \code{\link[torch]{torch_cholesky}}.

\code{chol} function is called by the function \code{\link{chol_solve}}.

For the \code{qr} decomposition see \code{\link{qr}}.

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


\code{chol} mimics the base function \code{chol} that computes Compute the Cholesky factorization of a real symmetric positive-definite square matrix. It returns a gpu.matrix-class object with The upper triangular factor of the Cholesky decomposition, i.e., the matrix \eqn{R} such that \eqn{R'R = X}.
}

\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(9),3,3)
ein <- eigen(a) #eigenvalues and eigenvectors
svd_return <- svd(a) #svd of gpu.matrix a

ata <- tcrossprod(a)
#ata is a real symmetric positive-definite square matrix.
chol(ata) #cholesky decomposition.


}
}
}

