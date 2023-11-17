\name{kroneker}
\title{kroneker Products}
\docType{methods}

\alias{\%x\%}
\alias{\%x\%-methods}
\alias{\%x\%,ANY,gpu.matrix.tensorflow-method}
\alias{\%x\%,ANY,gpu.matrix.torch-method}
\alias{\%x\%,gpu.matrix.tensorflow,ANY-method}
\alias{\%x\%,gpu.matrix.torch,ANY-method}

\description{
Kroneker product of two gpu.matrix-class objects. It mimics the base function \code{'kronecker'} to operate on gpu.matrix-class objects.
}

\details{
If gpu.matrix-class input is of type 'torch', the function internally call the function \code{\link[torch]{torch_kron}} of the library 'torch'
}

\usage{
\S4method{\%x\%}{ANY,gpu.matrix.tensorflow}(X,Y)
\S4method{\%x\%}{ANY,gpu.matrix.torch}(X,Y)
\S4method{\%x\%}{gpu.matrix.tensorflow,ANY}(X,Y)
\S4method{\%x\%}{gpu.matrix.torch,ANY}(X,Y)
}


\arguments{
\item{X}{A \code{\linkS4class{gpu.matrix}}.}
\item{Y}{A \code{\linkS4class{gpu.matrix}} or a matrix or a numeric variable}

}

\seealso{
See Also \code{\link[base]{kronecker}} and \code{\link[torch]{torch_kron}}.
}

\examples{

\donttest{
\dontrun{

a <- gpu.matrix(1:9,nrow=3,ncol=3)
a \%x\% diag(1,3)


}
}

}
