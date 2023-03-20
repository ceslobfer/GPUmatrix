\name{outer}
\title{outer}
\docType{methods}

\alias{\%o\%}
\alias{outer}

\alias{\%o\%methods}
\alias{outer-methods}
\alias{\%o\%,ANY,gpu.matrix.tensorflow-method}
\alias{\%o\%,ANY,gpu.matrix.torch-method}
\alias{\%o\%,gpu.matrix.tensorflow,ANY-method}
\alias{\%o\%,gpu.matrix.torch,ANY-method}

\alias{outer,ANY,gpu.matrix.tensorflow-method}
\alias{outer,ANY,gpu.matrix.torch-method}
\alias{outer,gpu.matrix.tensorflow,ANY-method}
\alias{outer,gpu.matrix.torch,ANY-method}




\description{
Equivalent outer product for a gpu.matrix. In mimics the base functions \code{outer} and \code{\%o\%}.
}


\usage{

\S4method{\%o\%}{ANY,gpu.matrix.tensorflow}(X,Y)
\S4method{\%o\%}{ANY,gpu.matrix.torch}(X,Y)
\S4method{\%o\%}{gpu.matrix.tensorflow,ANY}(X,Y)
\S4method{\%o\%}{gpu.matrix.torch,ANY}(X,Y)

\S4method{outer}{ANY,gpu.matrix.tensorflow}(X,Y,FUN,\dots)
\S4method{outer}{ANY,gpu.matrix.torch}(X,Y,FUN,\dots)
\S4method{outer}{gpu.matrix.tensorflow,ANY}(X,Y,FUN,\dots)
\S4method{outer}{gpu.matrix.torch,ANY}(X,Y,FUN,\dots)


}


\arguments{

\item{X,Y}{First and second arguments for function \code{FUN}. Tipicalle are a vector or an array. With this implementation can be also a gpu.matrix.}
\item{FUN}{a function to use on the outer products.}
\item{\dots}{Optional arguments to be passed to FUN.}

}

\seealso{
See Also \code{\link[base]{outer}}.
}




