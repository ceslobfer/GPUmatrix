\name{qr_decomposition}
\title{The QR Decomposition of a GPUmatrix object}
\description{
These functions mimic the base \code{qr} family functions to operate on gpu.matrix-class objects.

}

\alias{qr}
\alias{qr-methods}
\alias{qr,gpu.matrix.tensorflow-method}
\alias{qr,gpu.matrix.torch-method}

\alias{qr.Q}
\alias{qr.Q-methods}
\alias{qr.Q,list-method}

\alias{qr.R}
\alias{qr.R-methods}
\alias{qr.R,list-method}

\alias{qr.X}
\alias{qr.X-methods}
\alias{qr.X,list-method}

\alias{qr.coef}
\alias{qr.coef-methods}
\alias{qr.coef,list-method}

\alias{qr.qty}
\alias{qr.qty-methods}
\alias{qr.qty,list-method}

\alias{qr.qy}
\alias{qr.qy-methods}
\alias{qr.qy,list-method}

\alias{qr.resid}
\alias{qr.resid-methods}
\alias{qr.resid,list-method}


\alias{qr.solve}
\alias{qr.solve-methods}
\alias{qr.solve,ANY,gpu.matrix.tensorflow-method}
\alias{qr.solve,ANY,gpu.matrix.torch-method}
\alias{qr.solve,gpu.matrix.tensorflow,ANY-method}
\alias{qr.solve,gpu.matrix.tensorflow,gpu.matrix.tensorflow-method}
\alias{qr.solve,gpu.matrix.torch,ANY-method}
\alias{qr.solve,gpu.matrix.torch,gpu.matrix.torch-method}
\alias{qr.solve,list,ANY-method}

\usage{
\S4method{qr}{gpu.matrix.tensorflow}(x,\dots)
\S4method{qr}{gpu.matrix.torch}(x,\dots)

\S4method{qr.Q}{list}(qr,complete,Dvec)
\S4method{qr.R}{list}(qr,complete)
\S4method{qr.X}{list}(qr,complete)

\S4method{qr.coef}{list}(qr,y)
\S4method{qr.qy}{list}(qr,y)
\S4method{qr.qty}{list}(qr,y)
\S4method{qr.resid}{list}(qr,y)
\S4method{qr.solve}{ANY,gpu.matrix.tensorflow}(a,b)
\S4method{qr.solve}{ANY,gpu.matrix.torch}(a,b)
\S4method{qr.solve}{gpu.matrix.tensorflow,ANY}(a,b)
\S4method{qr.solve}{gpu.matrix.tensorflow,gpu.matrix.tensorflow}(a,b)
\S4method{qr.solve}{gpu.matrix.torch,ANY}(a,b)
\S4method{qr.solve}{gpu.matrix.torch,gpu.matrix.torch}(a,b)
\S4method{qr.solve}{list,ANY}(a,b)

}

\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}}.}
\item{y,b}{a \code{\linkS4class{gpu.matrix}} corresponding to the right-hand side of equations \code{ax=b} or \code{ax=y}.}
\item{\dots}{further arguments passed to or from other methods.}
\item{qr}{a list resulting from the application of the function \code{qr}.}
\item{complete}{The same as in 'base' function \code{qr.Q}, and \code{qr.X}}
\item{Dvec}{The same as in 'base' function \code{qr.Q}}
\item{a}{a \code{\linkS4class{gpu.matrix}} corresponding to the left-hand side of equations \code{ax=b} or \code{ax=y}.}


}

\details{
The function \code{qr} internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

The QR decomposition can be used to solve the equation \code{Ax=b} for a given matrix A, and a vector of observations b. In this context, the functions \code{qr.coef}, and \code{qr.resid} return the coefficients, and residuals values. Moreover, the functions \code{qr.qy}, and \code{qr.qty} returns \code{Q \%*\% y} and \code{Q \%*\% t(y)}.
Note that if parameter \code{complete} is TRUE then an arbitrary orthogonal completion of the \bold{X} and \bold{Q} matrix or wheter the \bold{R} matrix is to be completed by binding zero-value rows beneath the square upper triangle.

The function \code{solve.qr} solves the system of equations \code{Ax=b} via the QR decomposition. This function internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.
}

\value{
The function \code{qr} returns a list with the following items:
\item{q}{The corresponding complete matrix \code{Q} resulting from the application of the QR decomposition to \code{a}. It is a gpu.matrix-class object.}
\item{r}{The corresponding complete matrix \code{R} resulting from the application of the QR decomposition to \code{a}. It is a gpu.matrix-class object.}
\item{x}{The matrix \code{a}. It is a gpu.matrix-class object.}
Please note that the output returned by this function is different from the 'base' function \code{qr}, which returns an object of the 'qr' class.


After performing a QR decomposition on a matrix A, given the resulting object, the functions \code{qr.X}, \code{qr.Q}, and \code{qr.R} return the original matrix A, the matrix Q, and the matrix R respectively. The returned matrices are gpu.matrix-class objects.



The functions \code{qr.coef} and \code{qr.resid} return the coefficients and residuals when fitting the equation \code{Ax=b}. In this context, the  functions \code{qr.qy}, and \code{qr.qty} returns \code{Q \%*\% y} and \code{Q \%*\% t(y)}. The resulting vectors are objects of the class gpu.matrix.

The function \code{qr.solve} returns a gpu.matrix-class object containing the coefficients of the solution of the system of equations \code{Ax=b} by QR decomposition.



}


\seealso{
See \code{\link[base]{qr}}, \code{\link[torch]{linalg_qr}}, \code{\link[torch]{torch_triangular_solve}}
}

\examples{
\donttest{
\dontrun{
## overdetermined system
A <- gpu.matrix(runif(12),nrow =  4)
b <- gpu.matrix(rnorm(4),ncol=1)
qr.solve(a = A, b)
qr_gpu <- qr(A)
qr.solve(a=qr_gpu,b)
qr.coef(qr = qr_gpu,b)
qr.resid(qr = qr_gpu,b)
qr.qty(qr = qr_gpu,b)
qr.qy(qr = qr_gpu,b)
qr.X(qr = qr_gpu,complete = T)
qr.Q(qr = qr_gpu,complete = T)
qr.R(qr = qr_gpu,complete = T)


## underdetermined system
A <- gpu.matrix(runif(12),nrow =  3)
b <- gpu.matrix(rnorm(3),ncol=1)
qr.solve(a = A, b)
qr_gpu <- qr(A)
qr.solve(a=qr_gpu,b)
qr.coef(qr = qr_gpu,b)
qr.resid(qr = qr_gpu,b)
qr.qty(qr = qr_gpu,b)
qr.qy(qr = qr_gpu,b)
qr.X(qr = qr_gpu,complete = T)
qr.Q(qr = qr_gpu,complete = T)
qr.R(qr = qr_gpu,complete = T)
}
}
}
