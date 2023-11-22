\name{as_methods}
\title{as_methods}
\docType{methods}


\alias{as.array}
\alias{as.list}
\alias{as.matrix}
\alias{as.numeric}
\alias{as.vector}
\alias{is.numeric}


\alias{as.array-methods}
\alias{as.list-methods}
\alias{as.matrix-methods}
\alias{as.numeric-methods}
\alias{as.vector-methods}
\alias{is.numeric-methods}

\alias{as.array,gpu.matrix.tensorflow-method}
\alias{as.array,gpu.matrix.torch-method}
\alias{as.list,gpu.matrix.tensorflow-method}
\alias{as.list,gpu.matrix.torch-method}
\alias{as.matrix,gpu.matrix.tensorflow-method}
\alias{as.matrix,gpu.matrix.torch-method}
\alias{as.numeric,gpu.matrix.tensorflow-method}
\alias{as.numeric,gpu.matrix.torch-method}
\alias{as.vector,gpu.matrix.tensorflow-method}
\alias{as.vector,gpu.matrix.torch-method}
\alias{is.numeric,gpu.matrix.torch-method}
\alias{is.numeric,gpu.matrix.tensorflow-method}



\description{These functions mimic the 'base' functions of R that have the same name to operate on gpu.matrix-class objects:

Function \code{as.matrix} attempts to turn its argument into a matrix.
Function \code{as.list} attempts to turn its argument into a list.
Function \code{as.numeric} attempts to turn its argument into a numeric.
Function \code{as.array} attempts to turn its argument into an array.
Function \code{as.vector} attempts to turn its argument into a vector.
Function \code{is.numeric} is a general test of an object being interpretable as numbers.

}


\usage{

\S4method{as.array}{gpu.matrix.tensorflow}(x,\dots)
\S4method{as.array}{gpu.matrix.torch}(x,\dots)
\S4method{as.list}{gpu.matrix.tensorflow}(x,\dots)
\S4method{as.list}{gpu.matrix.torch}(x,\dots)
\S4method{as.matrix}{gpu.matrix.tensorflow}(x,\dots)
\S4method{as.matrix}{gpu.matrix.torch}(x,\dots)
\S4method{as.numeric}{gpu.matrix.tensorflow}(x,\dots)
\S4method{as.numeric}{gpu.matrix.torch}(x,\dots)
\S4method{as.vector}{gpu.matrix.tensorflow}(x,mode)
\S4method{as.vector}{gpu.matrix.torch}(x,mode)
\S4method{is.numeric}{gpu.matrix.torch}(x)
\S4method{is.numeric}{gpu.matrix.tensorflow}(x)

}


\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}} object.}
\item{\dots}{(generalized) vectors or matrices. These can be given as named arguments.}
\item{mode}{Argument for \code{as.vector}. It mimics the argument of the same name of the base function \code{as.vector}: character string naming an atomic mode or "list" or "expression" or (except for vector) "any".}

}


\value{
Given a gpu.matrix-class object:

Function \code{as.matrix} turns the input gpu.matrix to a 'matrix' object.

Function \code{as.list} turns the input gpu.matrix into a list.

Function \code{as.numeric} turns the input gpu.matrix into a numeric vector.

Function \code{as.array} turns the input gpu.matrix into an array (Since the gpu.matrix objects are always two-dimensional, this function is equivalent to \code{as.matrix}.).

Function \code{as.vector} turns the input gpu.matrix into a vector.

Function \code{is.numeric} returns TRUE or FAALSE if input can be interpretable as numbers.

}

\details{

Note that, if the input is a gpu.matrix with complex numbers: the function \code{is.numeric} will return FALSE, and the function \code{as.numeric} will only returns the real part and the following warning message: "In asMethod(object) : imaginary parts discarded in coercion".

The parameter \code{mode} of the function \code{as.vector} determines the storage mode of the result. For more details see \code{\link[base]{typeof}}.



}


\seealso{
\code{\link[base]{numeric}},
\code{\link[base]{array}},
\code{\link[base]{list}},
\code{\link[base]{matrix}},

}

\examples{
\donttest{
\dontrun{
a <- gpu.matrix(c(rnorm(8),2+1i),nrow=3,ncol=3)
as.array(a)
as.list(a)
as.matrix(a)
as.numeric(a)
is.numeric(a)
as.character(a)
as.vector(a,mode = "list")
as.vector(a,mode = "character")
as.vector(a,mode = "logical")
as.vector(a,mode = "integer")
as.vector(a,mode = "double")
as.vector(a,mode = "complex")
as.vector(a,mode = "raw")


}
}
}


