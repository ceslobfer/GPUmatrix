\name{solve_gpu.matrix}
\title{Solve a System of Equations}
\docType{methods}

\alias{solve}
\alias{solve-methods}
\alias{solve,ANY,gpu.matrix.tensorflow-method}
\alias{solve,ANY,gpu.matrix.torch-method}
\alias{solve,gpu.matrix.tensorflow,ANY-method}
\alias{solve,gpu.matrix.tensorflow,missing-method}
\alias{solve,gpu.matrix.torch,ANY-method}
\alias{solve,gpu.matrix.torch,missing-method}

\alias{ginv}
\alias{ginv-methods}
\alias{ginv,gpu.matrix.torch-method}
\alias{ginv,gpu.matrix.tensorflow-method}

\alias{chol_solve}
\alias{chol_solve-methods}
\alias{chol_solve,ANY,gpu.matrix.torch-method}
\alias{chol_solve,ANY,gpu.matrix.tensorflow-method}
\alias{chol_solve,gpu.matrix.torch,ANY-method}
\alias{chol_solve,gpu.matrix.tensorflow,ANY-method}


\description{
These functions are used to solve a system of equations or to compute the inverse of a matrix.

They mimic of the 'base' function \code{'solve'}: This generic function
solves the equation ax=b for x, where b can be either a vector or a matrix (gpu.matrix).

\code{ginv} mimics the function \code{ginv} of package MASS: it calculates the Moore-Penrose generalized inverse of a gpu.matrix.

\code{chol_solve} function is a GPUmatrix own function. This function uses the Cholesky decomposition to solve a system of equations.

}




\usage{

\S4method{solve}{ANY,gpu.matrix.tensorflow}(a,b)
\S4method{solve}{ANY,gpu.matrix.torch}(a,b)
\S4method{solve}{gpu.matrix.tensorflow,ANY}(a,b)
\S4method{solve}{gpu.matrix.tensorflow,missing}(a)
\S4method{solve}{gpu.matrix.torch,ANY}(a,b)
\S4method{solve}{gpu.matrix.torch,missing}(a)

\S4method{ginv}{gpu.matrix.torch}(X,tol)
\S4method{ginv}{gpu.matrix.tensorflow}(X,tol)

\S4method{chol_solve}{ANY,gpu.matrix.torch}(x,y)
\S4method{chol_solve}{ANY,gpu.matrix.tensorflow}(x,y)
\S4method{chol_solve}{gpu.matrix.torch,ANY}(x,y)
\S4method{chol_solve}{gpu.matrix.tensorflow,ANY}(x,y)


}


\arguments{

\item{a,x}{A square numeric or complex gpu.matrix containing the coefficients of the linear system. gpu.matrix that are logical matrices are coerced to numeric. For chol_solve, x must be the cholesky cecomposition of matrix a if a real symmetric positive-definite square gpu.matrix}

\item{b,y}{a numeric or complex vector or matrix giving the right-hand
                          side(s) of the linear system. If missing, b is taken to be an
                          identity matrix and solve will return the inverse of a.}

\item{X}{Matrix for which the Moore-Penrose inverse is required.}
\item{tol}{A relative tolerance to detect zero singular values.}

}

\seealso{
See also \code{\link[base]{solve}} and \code{\link[MASS]{ginv}}.

For cholesky decomposition see \code{\link[base]{chol}} from base or
 \code{\link[GPUmatrix]{matrix_decomposition}} from GPUmatrix.
}

\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
inv <- solve(a) #the inverse matrix
a \%*\% inv

b <- c(1,1,1)
betas <- solve(a,b)
a \%*\% betas


#inverse using ginv
inv_2 <- GPUmatrix::ginv(a)
a \%*\% inv_2


#chol_solve: it can be applies only if
# in the equation Ax=b A is real symmetric positive-definite square matrix.
a <- gpu.matrix(rnorm(9),3,3)
A <- tcrossprod(a) #A is symmetrix positive-definite
b <- gpu.matrix(rnorm(3))

x_solve <- solve(A,b) #using solve to compare results
x_chol_solve <- chol_solve(chol(A),b) #using chol_solve
#NOTE: notice that the input for chol_solve is the Cholesky decomposition
# of matrix A.

}
}
}

