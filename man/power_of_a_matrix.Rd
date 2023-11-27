\name{power_of_a_matrix}
\title{Compute the kth power of a matrix.}
\docType{methods}
\description{
Comput the kth power of a squere matrix, i.e., multiply the gpu.matrix-class object by itself as many times as user indicates.
}

\alias{\%^\%}
\alias{\%^\%,gpu.matrix.tensorflow,numeric-method}
\alias{\%^\%,gpu.matrix.torch,numeric-method}

\usage{
\S4method{\%^\%}{gpu.matrix.tensorflow,numeric}(x, k)
\S4method{\%^\%}{gpu.matrix.torch,numeric}(x, k)
}
%
\arguments{
  \item{x}{a \code{\linkS4class{gpu.matrix}}.}
  \item{k}{the power of the matrix.}
}
\details{
The input \code{x} gpu.matrix-class needs to be squere. This function internally call the method \code{\%*\%} as many times as required. If the input gpu.matrix-class object is stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.
}

\seealso{
See also: \code{\link{\%*\%}}.
}

\value{
the nth power of the input gpu.matrix-class object. The returned matrix is also a gpu.matrix-class object.
}
\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
a \%^\% 5


}
}
}
