\name{xtfrm}
\title{Auxiliary Function for Sorting and Ranking}
\docType{methods}

\description{
This function mimics the 'base' function \code{'xtfrm'} to operate on gpu.matrix-class objects.
}

\alias{xtfrm-methods}
\alias{xtfrm,gpu.matrix.tensorflow-method}
\alias{xtfrm,gpu.matrix.torch-method}

\usage{

\S4method{xtfrm}{gpu.matrix.tensorflow}(x)
\S4method{xtfrm}{gpu.matrix.torch}(x)

}


\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}}.}
}


\seealso{
\code{\link[base]{xtfrm}}
}

