\name{aperm}
\title{aperm}
\docType{methods}

\alias{aperm-methods}
\alias{aperm,gpu.matrix.tensorflow-method}
\alias{aperm,gpu.matrix.torch-method}


\description{
Transpose a GPUmatrix by permuting its dimensions and optionally resizing it.
}


\usage{

\S4method{aperm}{gpu.matrix.tensorflow}(a,perm,\dots)
\S4method{aperm}{gpu.matrix.torch}(a,perm,\dots)

}


\arguments{

\item{a}{A gpu.matrix.tensorflow o gpu.matrix.torch to be transposed}
\item{perm}{the subscript permutation vector, usually a permutation of the integers 1:n, where n is the number of dimensions of a.}
\item{\dots}{potential further arguments of methods.}

}


\seealso{
\code{\link[base]{aperm}}, \code{\link[GPUmatrix]{gpu.matrix}},
and \code{\link[GPUmatrix]{as.gpu.matrix}}.
}

\value{
It returns a transposed version of the gpu.matrix \code{a}, with subscripts permuted as indicated by the input \code{perm}. The output is a \code{matrix array} class object.

The resulting matrix can be easily converted to a \code{gpu.matrix} class object using the \code{gpu.matrix} or the \code{as.gpu.matrix} functions.
}


\examples{

\donttest{
\dontrun{

  #change the first with the second subscript.
  #this example corresponds to transposing the matrix
  a <- gpu.matrix(1:9,nrow=3,ncol=3,type = "tensorflow")
  b <- GPUmatrix::aperm(a,perm = c(2,1))


}
}

}


