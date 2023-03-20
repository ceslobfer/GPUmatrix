\name{diag}
\title{diag}

\description{

It mimics the function \code{mimics} that Extract or replace the diagonal of a matrix, or construct a diagonal matrix.

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

\item{x}{A gpu.matrix}
\item{value}{either a single value or a vector of length equal to that of the current diagonal.}
}


\seealso{
For more information see:

\code{\link[base]{diag}}

}

\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),3,3)

diag(a) #shows the diagonal of matrix a

diag(a) <- c(10,0,100) #set the diagonal of matrix a
a


}
}
}

