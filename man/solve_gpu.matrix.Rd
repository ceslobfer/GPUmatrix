\name{solve_gpu.matrix}
\title{solve_gpu.matrix}
\docType{methods}



\alias{solve-methods}
\alias{solve,ANY,gpu.matrix.tensorflow-method}
\alias{solve,ANY,gpu.matrix.torch-method}
\alias{solve,gpu.matrix.tensorflow,ANY-method}
\alias{solve,gpu.matrix.tensorflow,missing-method}
\alias{solve,gpu.matrix.torch,ANY-method}
\alias{solve,gpu.matrix.torch,missing-method}




\description{
Mimic of the base function \code{solve}: This generic function
solves the equation ax=b for x, where b can be either a vector or a matrix (gpu.matrix).
}


\usage{

\S4method{solve}{ANY,gpu.matrix.tensorflow}(a,b)
\S4method{solve}{ANY,gpu.matrix.torch}(a,b)
\S4method{solve}{gpu.matrix.tensorflow,ANY}(a,b)
\S4method{solve}{gpu.matrix.tensorflow,missing}(a)
\S4method{solve}{gpu.matrix.torch,ANY}(a,b)
\S4method{solve}{gpu.matrix.torch,missing}(a)

}


\arguments{

\item{a}{A square numeric or complex gpu.matrix containing the coefficients of the linear system.
                          gpu.matrix that are logical matrices are coerced to numeric.}

\item{b}{a numeric or complex vector or matrix giving the right-hand
                          side(s) of the linear system. If missing, b is taken to be an
                          identity matrix and solve will return the inverse of a.}

}

\seealso{
See also \code{\link[base]{solve}}.
}

\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),nrow=3,ncol=3,type="tensorflow")
inv <- solve(a) #the inverse matrix
a %*% inv

b <- c(1,1,1)
betas <- solve(a,b)
a%*%betas


}
}
}

