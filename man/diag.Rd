\name{diag}
\title{diag}
\description{
This function mimics the base function \code{'diag'} to operate on gpu.matrix-class objects: "extract or replace the diagonal of a matrix, or constructs a diagonal matrix."
}

\alias{diag}
\alias{diag-methods}
\alias{diag,gpu.matrix.tensorflow-method}
\alias{diag,gpu.matrix.torch-method}

\alias{diag<-}
\alias{diag<--methods}
\alias{diag<-,gpu.matrix.tensorflow,numeric-method}
\alias{diag<-,gpu.matrix.torch,numeric-method}

\usage{

\S4method{diag}{gpu.matrix.tensorflow}(x)
\S4method{diag}{gpu.matrix.torch}(x)
\S4method{diag}{gpu.matrix.tensorflow,numeric}(x) <- value
\S4method{diag}{gpu.matrix.torch,numeric}(x) <- value

}

\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}}.}
\item{value}{either a single value or a vector of length equal to that of the current diagonal.}
}

\value{
Output corresponding to the base function \code{diag}:
If input \code{x} is a gpu.matrix-class object then \code{diag{x}} returns a numeric object with the diagonal of the matrix \code{x} (this output is not a gpu.matrix-class object).

The replacement form \code{diag(x) <- value} sets the diagonal of the matrix \code{x} to the given value(s).
}

\seealso{
For more information see:
\code{\link[base]{diag}}

}

\examples{

if(installTorch()){

  a <- gpu.matrix(rnorm(9),nrow=3,ncol=3)

  diag(a) #shows the diagonal of matrix a

  diag(a) <- c(10,0,100) #set the diagonal of matrix a
  a

}





}

