\name{sort}
\title{sort}
\docType{methods}


\alias{sort}
\alias{sort-methods}
\alias{sort,gpu.matrix.tensorflow-method}
\alias{sort,gpu.matrix.torch-method}
\alias{sort,gpu.matrix.tensorflow,logical-method}
\alias{sort,gpu.matrix.torch,logical-method}




\description{
Mimics the 'base' function \code{'sort'}.
Given a 'GPUmatrix' object (partially), this function sort it into ascending or descending order.
}


\usage{

\S4method{sort}{gpu.matrix.tensorflow,logical}(x,decreasing)
\S4method{sort}{gpu.matrix.torch,logical}(x,decreasing)

}


\arguments{

\item{x}{A gpu.matrix.torch or a gpu.matrix.tensorflow object.}
\item{decreasing}{Logical. Should the sort be increasing or decreasing? }

}

\value{
Returns a vector of class numeric. In order to store the resultin vector in the GPU use the function as.gpu.matrix or the function gpu.matrix.
}

\seealso{
\code{\link[base]{sort}} and \code{\link{gpu.matrix}}
}


\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
sort(a) #returns a vector with the data sorted.
}
}
}

