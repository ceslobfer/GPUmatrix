\name{sort}
\title{sort}
\docType{methods}

\description{
This function mimics the 'base' function \code{sort} to operate on gpu.matrix-class objects.
This function sort the input matrix into ascending or descending order.
}


\alias{sort}
\alias{sort-methods}
\alias{sort,gpu.matrix.tensorflow-method}
\alias{sort,gpu.matrix.torch-method}
\alias{sort,gpu.matrix.tensorflow,logical-method}
\alias{sort,gpu.matrix.torch,logical-method}

\usage{

\S4method{sort}{gpu.matrix.tensorflow,logical}(x,decreasing)
\S4method{sort}{gpu.matrix.torch,logical}(x,decreasing)

}


\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}}.}
\item{decreasing}{Logical. Should the sort be increasing or decreasing? }

}

\details{
The function internally calls the corresponding function of the library torch or tensorflow (depending on the type of input gpu.matrix-class).

If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.


}

\value{
Returns a gpu.matrix-class object that is a vector (or a matrix with one column) with the values sorted.
}

\seealso{
\code{\link[base]{sort}}, \code{\link[torch]{torch_sort}}.
}


\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(9),nrow=3,ncol=3)
sort(a) #returns a vector with the data sorted.
}
}
}

