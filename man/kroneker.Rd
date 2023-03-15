\name{kroneker}
\title{kroneker}
\docType{methods}

\alias{\%x\%}
\alias{\%x\%-methods}
\alias{\%x\%,ANY,gpu.matrix.tensorflow-method}
\alias{\%x\%,ANY,gpu.matrix.torch-method}
\alias{\%x\%,gpu.matrix.tensorflow,ANY-method}
\alias{\%x\%,gpu.matrix.torch,ANY-method}

\description{
Kroneker for GPU matrices. Mimics the base function \code{kronecker}.
}


\usage{

\S4method{\%x\%}{ANY,gpu.matrix.tensorflow}(X,Y)
\S4method{\%x\%}{ANY,gpu.matrix.torch}(X,Y)
\S4method{\%x\%}{gpu.matrix.tensorflow,ANY}(X,Y)
\S4method{\%x\%}{gpu.matrix.torch,ANY}(X,Y)

}


\arguments{
\item{X}{A gpu.matrix}
\item{Y}{A gpu.matrix object or a matrix or a numeric variable}

}

\seealso{
See Also \code{\link[base]{kronecker}}.
}

\examples{

\donttest{
\dontrun{

a <- gpu.matrix(1:9,nrow=3,ncol=3)
a %x% diag(1,3)


}
}

}
