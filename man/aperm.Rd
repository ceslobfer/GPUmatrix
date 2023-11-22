\name{aperm}
\title{Array Transposition}
\docType{methods}

\description{
\code{t} returns the transpose of a gpu.matrix-class object.
}

\alias{t}
\alias{t-methods}
\alias{t,gpu.matrix.tensorflow-method}
\alias{t,gpu.matrix.torch-method}


\usage{

\S4method{t}{gpu.matrix.tensorflow}(x)
\S4method{t}{gpu.matrix.torch}(x)

}


\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}} to be transposed.}
}


\seealso{
For more information: \code{\link[base]{t}}.
}


\value{
It returns a transposed version of \code{a}. The output is also a \code{\linkS4class{gpu.matrix}} class object.}


\examples{

\donttest{
\dontrun{

  a <- gpu.matrix(1:9,nrow=3,ncol=3)
  t(a) #transpose of a.

}
}

}


