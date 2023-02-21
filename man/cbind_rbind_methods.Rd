\name{cbind_rbind_methods}
\title{cbind_rbind_methods}
\docType{methods}



\alias{cbind2-methods}
\alias{rbind2-methods}
\alias{cbind2,ANY,gpu.matrix.tensorflow-method}
\alias{cbind2,ANY,gpu.matrix.torch-method}
\alias{cbind2,gpu.matrix.tensorflow,ANY-method}
\alias{cbind2,gpu.matrix.torch,ANY-method}
\alias{rbind2,ANY,gpu.matrix.tensorflow-method}
\alias{rbind2,ANY,gpu.matrix.torch-method}
\alias{rbind2,gpu.matrix.tensorflow,ANY-method}
\alias{rbind2,gpu.matrix.torch,ANY-method}




\description{
Mimiks the cbind and rbind of base classes.
These functions take a sequence of vector, matrix or data-frame arguments and no GPUmatrix class
and combine by columns or rows, respectively.
}


\usage{

\S4method{cbind2}{ANY,gpu.matrix.tensorflow}(x,y)
\S4method{rbind2}{ANY,gpu.matrix.torch}(x,y)
\S4method{cbind2}{gpu.matrix.tensorflow,ANY}(x,y,\dots)
\S4method{rbind2}{gpu.matrix.torch,ANY}(x,y)

}


\arguments{

\item{x}{A gpu.matrix.tensorflow o gpu.matrix.torch object}
\item{y}{A gpu.matrix.tensorflow o gpu.matrix.torch object or any other matrix class}
\item{\dots}{(generalized) vectors or matrices. These can be given as named arguments.}

}
