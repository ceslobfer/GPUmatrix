\name{gpu.matrix}
\alias{gpu.matrix}
\alias{as.gpu.matrix}
\alias{as.gpu.matrix-methods}
\alias{as.gpu.matrix,ANY-method}
\title{
create and store a matrix in the GPU
}
\description{
Mimic the base \code{matrix} function.

gpu.matrix creates a gpu.matrix.torch or gpu.matrix.tensorflow from input data. The created matrix is stored in the GPU. This function also mimics the function 'Matrix' of the library 'Matrix'.
}
\usage{
gpu.matrix(data = NA, nrow = NULL, ncol = NULL, byrow = FALSE,
                       dimnames = NULL, dtype=NULL, sparse=NULL, colnames=c(),
                       rownames=c(),device="cuda", type="torch")

as.gpu.matrix(x,\dots)
\S4method{as.gpu.matrix}{ANY}(x,\dots)
}
\arguments{
  \item{data,x}{
a scalar, vector or matrix (both matrix or Matrix class).
}
  \item{nrow}{
Number of rows of the matrix. By default the number of rows of data if data is a matrix or a Matrix.
}
  \item{ncol}{
Number of columns of the matrix. By default the number of columns of data if data is a matrix or a Matrix.
}
  \item{byrow}{
The same as  function 'matrix': "logical. If FALSE (the default) the matrix is filled by columns, otherwise the matrix is filled by rows."
}
  \item{dimnames}{
The same as in  function 'matrix': "A dimnames attribute for the matrix: NULL or a list of length 2 giving the row and column names respectively. An empty list is treated as NULL, and a list of length one as row names. The list can be named, and the list names will be used as names for the dimensions."
}
  \item{dtype}{
By default "float64". User can indicate "float32" or "int" for "int64".
}
  \item{sparse}{
The same as in function 'Matrix' of the library 'Matrix': "logical or NULL, specifying if the result should be sparse or not. By default, it is made sparse when more than half of the entries are 0."
}
  \item{colnames}{
A vector with the column names.
}
  \item{rownames}{
A vector with the row names.
}
\item{type}{If gpu.matrix is "torch"(default) or "tensorflow."}
\item{device}{By deafult "cuda". indicates de devide to load cuda.}
\item{\dots}{additional arguments to be passed to or from methods.}

}
\details{
gpu.matrix.tensorflow mimics the functions Matrix of the library Matrix and the base function matrix. If tensorflow and/or torch are properly installed then the gpu.matrix will be stored on the GPU. The example shows how to check this.

User can apply -using the same operator- to this matrix the basic functions that can be applied to a matrix of class 'matrix' or class 'Matrix'.

It can work also with sparse matrices as the library Matrix.
}
\value{
Returns matrix of class 'gpu.matrix' of the library GPUmatrix that can be either "gpu.matrix.tensorflow" or ""gpu.matrix.torch". For both torch and tensorflow the functions to be applied to a matrix are the same.

If the gpu.matrix is not sparse it will show on the console the matrix as it is. If the gpu.matrix is sparse, it will return to the console the position where there are number different from zero. The internal values of the matrix can be seen using the operator "@".

Even if the matrix is sparse or not, both kind of matrices works equally with all functions.
}
\author{
Cesar Lobato and Angel Rubio.
}
\seealso{
See \code{\linkS4class{Matrix}} and \code{\linkS4class{matrix}}.
}
\examples{

\donttest{
\dontrun{
## create a gpu.matrix.tensorflow and check it is stored in the GPU.
a <- gpu.matrix(1:9,nrow=3,ncol=3,type="tensorflow")
class(a)
a@gm$device

# the output of class(a) should be:
#[1] "gpu.matrix.tensorflow"
#attr(,"package")
#[1] "GPUmatrix"

#the output of a@gm@device should have a similar shape:
#[1] "/job:localhost/replica:0/task:0/device:GPU:0"
}
}
}

