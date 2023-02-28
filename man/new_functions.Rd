\name{new_functions}
\title{new_functions}
\docType{methods}


\alias{as.gpu.matrix-methods}
\alias{cor-methods}
\alias{determinant-methods}
\alias{diag<--methods}
\alias{dim<--methods}
\alias{dimnames<--methods}
\alias{expmGPU-methods}
\alias{round-methods}
\alias{as.gpu.matrix,ANY-method}
\alias{cor,gpu.matrix.tensorflow,ANY,ANY,ANY-method}
\alias{cor,gpu.matrix.tensorflow,ANY,missing,character-method}
\alias{cor,gpu.matrix.torch,ANY,ANY,ANY-method}
\alias{cor,gpu.matrix.torch,ANY,missing,character-method}
\alias{determinant,gpu.matrix.tensorflow,logical-method}
\alias{determinant,gpu.matrix.tensorflow,missing-method}
\alias{determinant,gpu.matrix.torch,logical-method}
\alias{determinant,gpu.matrix.torch,missing-method}
\alias{diag<-,gpu.matrix.tensorflow,numeric-method}
\alias{diag<-,gpu.matrix.torch,numeric-method}
\alias{dim<-,gpu.matrix.tensorflow,vector-method}
\alias{dim<-,gpu.matrix.torch,vector-method}
\alias{dimnames<-,gpu.matrix.tensorflow,vector-method}
\alias{dimnames<-,gpu.matrix.torch,vector-method}
\alias{expmGPU,gpu.matrix.tensorflow-method}
\alias{round,gpu.matrix.tensorflow,ANY-method}
\alias{round,gpu.matrix.torch,missing-method}
\alias{round,gpu.matrix.torch,numeric-method}




\description{
Mimic of functions applied to matrix or Matrix class objects.
}


\usage{
\S4method{as.gpu.matrix}{ANY}(x,\dots)
\S4method{cor}{gpu.matrix.tensorflow,ANY,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.tensorflow,ANY,missing,character}(x,y,method)
\S4method{cor}{gpu.matrix.torch,ANY,ANY,ANY}(x,y)
\S4method{cor}{gpu.matrix.torch,ANY,missing,character}(x,y,method)
\S4method{determinant}{gpu.matrix.tensorflow,logical}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.tensorflow,missing}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.torch,logical}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.torch,missing}(x,logarithm,\dots)
\S4method{diag}{gpu.matrix.tensorflow,numeric}(x) <- value
\S4method{diag}{gpu.matrix.torch,numeric}(x) <- value
\S4method{dim}{gpu.matrix.tensorflow,vector}(x) <- value
\S4method{dim}{gpu.matrix.torch,vector}(x) <- value
\S4method{dimnames}{gpu.matrix.tensorflow,vector}(x) <- value
\S4method{dimnames}{gpu.matrix.torch,vector}(x) <- value
\S4method{expmGPU}{gpu.matrix.tensorflow}(x)
\S4method{round}{gpu.matrix.tensorflow,ANY}(x)
\S4method{round}{gpu.matrix.torch,missing}(x,digits)
\S4method{round}{gpu.matrix.torch,numeric}(x,digits)

}


\arguments{

\item{x}{any matrix class}
\item{\dots}{extra generic parameters of gpu.matrix function}
\item{y}{gput.matrix or other type of matrix}
\item{method}{a character string indicating which correlation
    coefficient (or covariance) is to be computed.
    One of pearson' (default), 'kendall',
    or 'spearman': can be abbreviated.}
\item{logarithm}{logical; if TRUE (default) return the logarithm of the modulus of the determinant.}
\item{value}{either a single value or a vector of length equal to that of the current diagonal.}
\item{digits}{integer indicating the number of decimal places (round) or significant digits (signif) to be used.}

}
