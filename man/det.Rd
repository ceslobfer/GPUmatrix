\name{det}
\title{Calculate the Determinant of a 'GPUMatrix'}
\description{
These functions mimic the 'base' functions \code{det} and \code{determinant} to operate on gpu.matrix-class objects: "\code{det} calculates the determinant of a matrix. \code{determinant} is a generic function that returns separately the modulus of the determinant, optionally on the logarithm scale, and the sign of the determinant."

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

\item{x}{a \code{\linkS4class{gpu.matrix}} object.}
\item{\dots}{Optional parameters. For more details seee \code{\link[base]{det}}}
\item{logarithm}{logical; if TRUE (default) return the logarithm of the modulus of the determinant.}

}

\seealso{
For more information see:
\code{\link[base]{det}}.
}

\details{

The function \code{det} and \code{determinant} internally call the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.
}



\value{

\code{det} returns the same output corresponding to the base function \code{det}, which is the determinant of \code{x}. The returned value is a object of class numeric sotred in the cpu.

\code{determinant} returns the corresponding output of the base function \code{determinant}, which is an object of class \code{det}, that contains the following components:

\item{modulus}{a numeric value. The modulus (absolute value) of the determinant if logarithm is FALSE; otherwise the logarithm of the modulus.}
\item{sign}{integer; either +1 or -1 according to whether the determinant is positive or negative.}
}


\examples{
\donttest{
\dontrun{

x <- gpu.matrix(1:4,nrow=2, ncol = 2)
determinant(x) #modulus of the determinant.
det(x)#the determinant.

}
}
}


