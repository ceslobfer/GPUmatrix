\name{as_methods}
\title{as_methods}
\docType{methods}



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




\description{
as.matrix attempts to turn its argument into a matrix.
as.list attempts to turn its argument into a list.
as.numeric attempts to turn its argument into a numeric.
as.vector attempts to turn its argument into a vector.
is.numeric is a general test of an object being interpretable as numbers
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

}


\arguments{

\item{x}{A gpu.matrix.tensorflow o gpu.matrix.torch object}
\item{\dots}{(generalized) vectors or matrices. These can be given as named arguments.}
\item{mode}{character string naming an atomic mode or "list" or "expression" or (except for vector) "any".}

}


\seealso{
\code{\link[base]{numeric}},
\code{\link[base]{array}},
\code{\link[base]{list}},
\code{\link[base]{matrix}},

}
