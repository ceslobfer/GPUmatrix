\name{matrix-product}
\title{Matrix Products}
\docType{methods}

\alias{\%*\%}
\alias{\%*\%-methods}
\alias{\%*\%,ANY,gpu.matrix.tensorflow-method}
\alias{\%*\%,ANY,gpu.matrix.torch-method}
\alias{\%*\%,gpu.matrix.tensorflow,ANY-method}
\alias{\%*\%,gpu.matrix.torch,ANY-method}

\alias{crossprod}
\alias{crossprod-methods}
\alias{crossprod,ANY,gpu.matrix.tensorflow-method}
\alias{crossprod,gpu.matrix.tensorflow,ANY-method}
\alias{crossprod,gpu.matrix.tensorflow,missing-method}
\alias{crossprod,ANY,gpu.matrix.torch-method}
\alias{crossprod,gpu.matrix.torch,ANY-method}
\alias{crossprod,gpu.matrix.torch,missing-method}


\alias{tcrossprod}
\alias{tcrossprod-methods}
\alias{tcrossprod,ANY,gpu.matrix.tensorflow-method}
\alias{tcrossprod,gpu.matrix.tensorflow,ANY-method}
\alias{tcrossprod,gpu.matrix.tensorflow,missing-method}
\alias{tcrossprod,ANY,gpu.matrix.torch-method}
\alias{tcrossprod,gpu.matrix.torch,ANY-method}
\alias{tcrossprod,gpu.matrix.torch,missing-method}



\description{
 Mimics the base function for matrix multiplication.
 Given the matrices \code{x} and \code{y} as inputs, return a matrix cross-product. The input \code{x} and \code{y} can be of class gpu.matrix. Thus, the operation will be performed by the GPU.
}
\usage{
\S4method{\%*\%}{gpu.matrix.tensorflow,ANY}(x, y)
\S4method{\%*\%}{gpu.matrix.torch,ANY}(x, y)

\S4method{crossprod}{gpu.matrix.tensorflow,ANY}(x, y,\dots)
\S4method{crossprod}{gpu.matrix.tensorflow,missing}(x, y = NULL,\dots)

\S4method{tcrossprod}{gpu.matrix.tensorflow,ANY}(x, y,\dots)
\S4method{tcrossprod}{gpu.matrix.tensorflow,missing}(x, y = NULL,\dots)


\S4method{crossprod}{gpu.matrix.torch,ANY}(x, y,\dots)
\S4method{crossprod}{gpu.matrix.torch,missing}(x, y = NULL,\dots)

\S4method{tcrossprod}{gpu.matrix.torch,ANY}(x, y,\dots)
\S4method{tcrossprod}{gpu.matrix.torch,missing}(x, y = NULL,\dots)

}
\arguments{
  \item{x}{a gpu.matrix}
  \item{y}{a gpu.matrix or matrix or Matrix object, or for \code{[t]crossprod()}
    \code{NULL} (by default); the latter case is formally equivalent to
    \code{y = x}.}
    \item{\dots}{potentially more arguments passed to and from methods.}
}
\section{Methods}{
  \describe{
    \item{\%*\%}{\code{signature(x = "gpu.matrix.tensorflow", y = "ANY")}:
    Matrix multiplication}

    \item{crossprod}{\code{signature(x = "gpu.matrix.tensorflow", y = "ANY")}:
    Matrix multiplication}

    \item{crossprod}{\code{signature(x = "gpu.matrix.tensorflow", y = "missing")}:
    Matrix multiplication}

    \item{tcrossprod}{\code{signature(x = "gpu.matrix.tensorflow", y = "ANY")}:
    Matrix multiplication}

    \item{tcrossprod}{\code{signature(x = "gpu.matrix.tensorflow", y = "missing")}:
    Matrix multiplication}

    \item{\%*\%}{\code{signature(x = "gpu.matrix.torch", y = "ANY")}:
    Matrix multiplication}

    \item{crossprod}{\code{signature(x = "gpu.matrix.torch", y = "ANY")}:
    Matrix multiplication}

    \item{crossprod}{\code{signature(x = "gpu.matrix.torch", y = "missing")}:
    Matrix multiplication}

    \item{tcrossprod}{\code{signature(x = "gpu.matrix.torch", y = "ANY")}:
    Matrix multiplication}

    \item{tcrossprod}{\code{signature(x = "gpu.matrix.torch", y = "missing")}:
    Matrix multiplication}

    }

}
\seealso{
  \code{\link[base]{tcrossprod}} in \R's base, and
  \code{\link{crossprod}} and \code{\link{\%*\%}}.
  \pkg{Matrix} package \code{\link{\%&\%}} for boolean matrix product
  methods.
}
\examples{
\donttest{
\dontrun{
  a <- gpu.matrix.tensorflow(as.numeric(1:9),nrow=3,ncol=3)
  b <- a\%*\%a

  b <- tcrossprod(a)

  b <- crossprod(a,a)

}
}
}
