\name{det}
\title{Calculate the Determinant of a Matrix}


\description{

Mimic of the base functions \code{det} and \code{determinant}: \code{det} calculates the determinant of a gpu.matrix. \code{determinant} is a generic function that returns separately the modulus of the determinant, optionally on the logarithm scale, and the sign of the determinant.

}


\alias{determinant-methods}
\alias{determinant,gpu.matrix.tensorflow,logical-method}
\alias{determinant,gpu.matrix.tensorflow,missing-method}
\alias{determinant,gpu.matrix.torch,logical-method}
\alias{determinant,gpu.matrix.torch,missing-method}

\alias{det-methods}
\alias{det,gpu.matrix.tensorflow-method}
\alias{det,gpu.matrix.torch-method}


\usage{

\S4method{determinant}{gpu.matrix.tensorflow,logical}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.tensorflow,missing}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.torch,logical}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.torch,missing}(x,logarithm,\dots)

\S4method{det}{gpu.matrix.tensorflow}(x,\dots)
\S4method{det}{gpu.matrix.torch}(x,\dots)

}


\arguments{

\item{x}{numeric gpu.matrix}
\item{\dots}{extra generic parameters of gpu.matrix function}
\item{logarithm}{logical; if TRUE (default) return the logarithm of the modulus of the determinant.}

}

\seealso{
For more information see:
\code{\link[base]{det}}
}


\examples{
\donttest{
\dontrun{

x <- gpu.matrix(1:4,nrow=2, ncol = 2,type="tensorflow")
unlist(determinant(x)) #modulus of the determinant.
det(x)#the determinant.

}
}
}


