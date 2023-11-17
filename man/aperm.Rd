\name{aperm}
\title{Array Transposition}
\docType{methods}

\description{
\code{aperm} transposes a gpu.matrix-class object by permuting its dimensions. GPUmatrix package only works with matrices of two dimensions. Therefore, this function is equivalent to the function \code{t}.

\code{t} returns the transpose of a gpu.matrix-class object.
}

\alias{aperm}
\alias{t}

\alias{aperm-methods}
\alias{aperm,gpu.matrix.tensorflow-method}
\alias{aperm,gpu.matrix.torch-method}
\alias{t-methods}
\alias{t,gpu.matrix.tensorflow-method}
\alias{t,gpu.matrix.torch-method}


\usage{

\S4method{aperm}{gpu.matrix.tensorflow}(a,perm,\dots)
\S4method{aperm}{gpu.matrix.torch}(a,perm,\dots)

\S4method{t}{gpu.matrix.tensorflow}(x)
\S4method{t}{gpu.matrix.torch}(x)

}


\arguments{

\item{x,a}{a \code{\linkS4class{gpu.matrix}} to be transposed.}
\item{perm}{the subscript permutation vector, usually a permutation of the integers 1:n, where n is the number of dimensions of a. In GPUmatrix, the maximum value of n is 2 as GPUmatrix only works with matrices of 2 dimensions.}
\item{\dots}{potential further arguments of methods.}

}


\seealso{
For more information:
\code{\link[base]{aperm}}, \code{\link[base]{t}},  \code{\link[GPUmatrix]{gpu.matrix}},
and \code{\link[GPUmatrix]{as.gpu.matrix}}.
}

\details{
The base function \code{\link[base]{aperm}} includes a third parameter "resize" that indicates whether the matrix should be resized or not. For GPUmatrix we considered this parameter to be always TRUE.
}

\value{
It returns a transposed version of \code{a}, with subscripts permuted as indicated by the input \code{perm}. The output is also a \code{\linkS4class{gpu.matrix}} class object.}


\examples{

\donttest{
\dontrun{

  #change the first with the second subscript.
  a <- gpu.matrix(1:9,nrow=3,ncol=3)
  b <- aperm(a,perm = c(2,1))

  #the same can be done using the function t:
  t(a) #transpose of a.

}
}

}


