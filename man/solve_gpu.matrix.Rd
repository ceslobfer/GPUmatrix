\name{solve_gpu.matrix}
\title{Solve a System of Equations}
\docType{methods}

\description{

The function \code{solve} mimics of the 'base' function \code{solve} to operate on gpu.matrix-class objects: it "solves the equation \code{a \%*\% x = b}."

The function \code{ginv} mimics the function \code{ginv} of package 'MASS' to operate on gpu.matrix-class objects: it "Calculates the Moore-Penrose generalized inverse of a matrix X."

The function \code{chol_solve} is a GPUmatrix own function. This function uses the Cholesky decomposition to solve a system of equations.

}

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

These inputs correspond to the \code{solve} function:

\item{a}{a square numeric or complex \code{\linkS4class{gpu.matrix}} containing the coefficients of the linear system.}

\item{b}{a numeric or complex vector or matrix giving the right-hand side(s) of the linear system. If \code{b} missing, \code{solve} will return the inverse of \code{a}.}


These inputs correspond to the \code{chol_solve} function:

\item{x}{Given the equation \code{Ax=b}, \code{x} must be the transponsed of the cholesky decomposition of matrix \code{A} if \code{A} is a real symmetric positive-definite square matrix.}

\item{y}{a numeric or complex vector or matrix giving the right-hand side(s) of the linear system.}

These inputs correspond to the \code{ginv} function:

\item{X}{Matrix for which the Moore-Penrose inverse is required.}
\item{tol}{A relative tolerance to detect zero singular values.}

}

\details{
The functions \code{solve}, and \code{ginv} internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.

}

\value{
The result of these functions is  an object of the class gpu.matrix.
}



\seealso{
See also \code{\link[base]{solve}}, \code{\link[MASS]{ginv}}, \code{\link[torch]{torch_inverse}}, and \code{\link[torch]{torch_pinverse}}.

For cholesky decomposition see \code{\link[base]{chol}} from base or
 \code{\link[GPUmatrix]{matrix_decomposition}} from GPUmatrix.

Also see \code{\link{qr.solve}}.

}

\examples{
\donttest{
\dontrun{

#solve a system of equations:
a <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
b <- c(1,1,1)
betas <- solve(a,b)
a \%*\% betas

#the inverse matrix
inv <- solve(a)
a \%*\% inv

#inverse using ginv
inv_2 <- ginv(a)
a \%*\% inv_2


#chol_solve: it can be applies only if
# in the equation Ax=b A is real symmetric positive-definite square matrix.
a <- gpu.matrix(rnorm(9),3,3)
A <- tcrossprod(a) #A is symmetrix positive-definite
b <- gpu.matrix(rnorm(3))

x_solve <- solve(A,b) #using solve to compare results
x_chol_solve <- chol_solve(t(chol(A)),b) #using chol_solve
#NOTE: notice that the input for chol_solve is the Cholesky decomposition
# of matrix A.

}
}
}

