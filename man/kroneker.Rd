\name{kroneker}
\title{kroneker Products}
\docType{methods}

\description{
Kroneker product of two gpu.matrix-class objects. This function mimics the 'base' function \code{'kronecker'} to operate on gpu.matrix-class objects.
}


\alias{\%x\%}
\alias{\%x\%-methods}
\alias{\%x\%,ANY,gpu.matrix.tensorflow-method}
\alias{\%x\%,ANY,gpu.matrix.torch-method}
\alias{\%x\%,gpu.matrix.tensorflow,ANY-method}
\alias{\%x\%,gpu.matrix.torch,ANY-method}


\details{
The function \code{\%x\%} internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}

\usage{
\S4method{\%x\%}{ANY,gpu.matrix.tensorflow}(X,Y)
\S4method{\%x\%}{ANY,gpu.matrix.torch}(X,Y)
\S4method{\%x\%}{gpu.matrix.tensorflow,ANY}(X,Y)
\S4method{\%x\%}{gpu.matrix.torch,ANY}(X,Y)
}


\arguments{
\item{X}{A \code{\linkS4class{gpu.matrix}}.}
\item{Y}{A \code{\linkS4class{gpu.matrix}} or a matrix or a numeric variable.}

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
