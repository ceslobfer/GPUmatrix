\name{matrix-product}
\title{Matrix Products}
\docType{methods}

\description{
Mimic of the 'base' functions \code{\%*\%}, \code{crossprod}, \code{tcrossprod} to operate on gpu.matrix-class objects.
}


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
  \item{x}{a \code{\linkS4class{gpu.matrix}}.}
  \item{y}{a \code{\linkS4class{gpu.matrix}}, 'matrix' or 'Matrix' object. For the functions \code{tcrossprod} and \code{crossprod} is NULL (by default), that is equivalent to \code{x=y}.}
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
  Also see \code{\link[torch]{torch_matmul}}
}

\details{

Internally, these functions call the appropriate tensorflow or torch function to perform the matrix product (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.

}


\value{
A gpu.matrix-class object with the result of the matrix product.
}


\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(12),nrow=4,ncol=3)
b <- t(a)%*%a
b
crossprod(a,a)

b <- a%*%t(a)
b
tcrossprod(a)


}
}
}
