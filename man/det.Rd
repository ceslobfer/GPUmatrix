\name{det}
\title{Calculate the Determinant of a 'GPUMatrix'}


\description{

Mimic of the base functions \code{'det'} and \code{'determinant'}: \code{'det'} calculates the determinant of a gpu.matrix. \code{'determinant'} is a generic function that returns separately the modulus of the determinant, optionally on the logarithm scale, and the sign of the determinant.

}

\alias{determinant}
\alias{determinant-methods}
\alias{determinant,gpu.matrix.tensorflow,logical-method}
\alias{determinant,gpu.matrix.tensorflow,missing-method}
\alias{determinant,gpu.matrix.torch,logical-method}
\alias{determinant,gpu.matrix.torch,missing-method}

\alias{det}
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

\value{
\code{det} returns the same output corresponding to the base function 'det', which is the determinant of \code{x}.

\code{determinant} returns the corresponding output of the base function 'determinant', which is a list with components:

\item{modulus}{a numeric value. The modulus (absolute value) of the determinant if logarithm is FALSE; otherwise the logarithm of the modulus.}
\item{sign}{integer; either +1 or -1 according to whether the determinant is positive or negative.}
}


\examples{
\donttest{
\dontrun{

x <- gpu.matrix(1:4,nrow=2, ncol = 2)
unlist(determinant(x)) #modulus of the determinant.
det(x)#the determinant.

}
}
}


