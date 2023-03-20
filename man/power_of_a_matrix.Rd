\name{power_of_a_matrix}
%
\title{Compute the kth power of a matrix.}
\docType{methods}
\alias{\%^\%}
\alias{\%^\%,gpu.matrix.tensorflow,numeric-method}
\alias{\%^\%,gpu.matrix.torch,numeric-method}
\description{
Comput the kth power of a squere matrix, i.e., multiply the matrix by itself as many times as user indicates.
}
\usage{
\S4method{\%^\%}{gpu.matrix.tensorflow,numeric}(x, k)
\S4method{\%^\%}{gpu.matrix.torch,numeric}(x, k)
}
%
\arguments{
  \item{x}{a gpu.matrix.tensorflow or gput.matrix.torch}
  \item{k}{the power of the matrix}
}
\details{
x needs to be a squere matrix.
}
\value{
the nth power of the input gpu.matrix. The returned matrix is also a gpu.matrix.
}
\examples{
\donttest{
\dontrun{

a <- gpu.matrix(1:9,nrow=3,ncol=3)
a \%^\% 5


}
}
}
