\name{new_functions}
\title{new_functions}
\docType{methods}






\alias{determinant-methods}
\alias{diag<--methods}
\alias{round-methods}
\alias{determinant,gpu.matrix.tensorflow,logical-method}
\alias{determinant,gpu.matrix.tensorflow,missing-method}
\alias{determinant,gpu.matrix.torch,logical-method}
\alias{determinant,gpu.matrix.torch,missing-method}
\alias{diag<-,gpu.matrix.tensorflow,numeric-method}
\alias{diag<-,gpu.matrix.torch,numeric-method}
\alias{round,gpu.matrix.tensorflow,ANY-method}
\alias{round,gpu.matrix.torch,missing-method}
\alias{round,gpu.matrix.torch,numeric-method}
\alias{round,gpu.matrix.tensorflow,missing-method}
\alias{round,gpu.matrix.tensorflow,numeric-method}




\description{
Mimic of functions applied to matrix or Matrix class objects.
}


\usage{
\S4method{determinant}{gpu.matrix.tensorflow,logical}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.tensorflow,missing}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.torch,logical}(x,logarithm,\dots)
\S4method{determinant}{gpu.matrix.torch,missing}(x,logarithm,\dots)
\S4method{diag}{gpu.matrix.tensorflow,numeric}(x) <- value
\S4method{diag}{gpu.matrix.torch,numeric}(x) <- value

\S4method{round}{gpu.matrix.tensorflow,ANY}(x)
\S4method{round}{gpu.matrix.torch,missing}(x,digits)
\S4method{round}{gpu.matrix.torch,numeric}(x,digits)
\S4method{round}{gpu.matrix.tensorflow,missing}(x,digits)
\S4method{round}{gpu.matrix.tensorflow,numeric}(x,digits)

}


\arguments{

\item{x}{any matrix class}
\item{\dots}{extra generic parameters of gpu.matrix function}
\item{logarithm}{logical; if TRUE (default) return the logarithm of the modulus of the determinant.}
\item{value}{either a single value or a vector of length equal to that of the current diagonal.}
\item{digits}{integer indicating the number of decimal places (round) or significant digits (signif) to be used.}

}
