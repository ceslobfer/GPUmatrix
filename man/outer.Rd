\name{outer}
\title{outer}
\docType{methods}



\alias{\%o\%-methods}
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
Equivalent outer producto for GPUmatrix
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

\item{X}{A gpu.matrix.tensorflow o gpu.matrix.torch object.}
\item{Y}{A gpu.matrix.tensorflow o gpu.matrix.torch object or a matrix or a numeric variable.}
\item{FUN}{a function to use on the outer products.}
\item{\dots}{ptional arguments to be passed to FUN.}

}
