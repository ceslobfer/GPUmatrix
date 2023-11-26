\name{type of gpu.matrix}
\title{Spicify type of 'GPUmatrix'}
\docType{methods}


\description{

\code{dtype} and \code{dtype<-} are functions that show or set the number of bits to use to store the number. The possible options are "float64" for float64 (default), "float32" for float32 and "int" for int64.
float64 uses 64 bits, that means that float64's take up twice as much memory thatn float32, thus doing operations on them may be slower in some machine architectures. However, float64's can represent numbers much more accurately than 32 bit floats. They also allow much larger numbers to be stored.

\code{to_dense} is a function that transforms a sparse matrix to a dense matrix. On the other hand, \code{to_sparse} transforms a dense matrix to a sparse matrix.

}



\alias{dtype}
\alias{dtype<-}
\alias{dtype-methods}
\alias{dtype<--methods}
\alias{dtype,gpu.matrix.torch-method}
\alias{dtype,gpu.matrix.tensorflow-method}
\alias{dtype<-,gpu.matrix.torch-method}
\alias{dtype<-,gpu.matrix.tensorflow-method}

\alias{to_dense}
\alias{to_sparse}
\alias{to_dense-methods}
\alias{to_sparse-methods}
\alias{to_dense,gpu.matrix.torch-method}
\alias{to_dense,gpu.matrix.tensorflow-method}
\alias{to_sparse,gpu.matrix.torch-method}
\alias{to_sparse,gpu.matrix.tensorflow-method}


\usage{

\S4method{to_dense}{gpu.matrix.torch}(x)
\S4method{to_dense}{gpu.matrix.tensorflow}(x)
\S4method{to_sparse}{gpu.matrix.torch}(x)
\S4method{to_sparse}{gpu.matrix.tensorflow}(x)


\S4method{dtype}{gpu.matrix.torch}(x)
\S4method{dtype}{gpu.matrix.tensorflow}(x)
\S4method{dtype}{gpu.matrix.torch}(x) <- value
\S4method{dtype}{gpu.matrix.tensorflow}(x) <- value

}


\arguments{

\item{x}{a \code{\linkS4class{gpu.matrix}}.}
\item{value}{type of gpu.matrix object}

}


\value{
\code{dtype} and \code{dtype <- } show or set the number of bits to use to store the number.

\code{to_dense} returns a dense gpu.matrix-class object while the function \code{to_sparse} returns a sparse gpu.matrix-class object.

}


\seealso{
See also \code{\link[GPUmatrix]{gpu.matrix}}.
}

\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(9),3,3)

dtype(a) #bits used to store the numbers: it is float64 by default.

b <- a
dtype(b) <- "float32" #change to float32
b

b <- a
dtype(b) <- "int" #change to integer64 (int64)
b

#sparse or dense matrices
A <- gpu.matrix(data=c(1,1,1,0,0,1,0,1,0),3,3)
A #A is a dense gpu.matrix

A_sparse <- to_sparse(A) #transform A to a sparse matrix.
A_sparse #this matrix stores the where number different to 0 were placed.

to_dense(A_sparse) #transform A_sparse to a dense matrix and we obtain the orginal matrix A:
A

}
}


}

