\name{gpu.matrix}
\alias{gpu.matrix}
\alias{as.gpu.matrix}
\alias{as.gpu.matrix-methods}
\alias{as.gpu.matrix,ANY-method}
\title{create and store a matrix in the GPU}
\description{
Mimic the base \code{'matrix'} function to create a gpu.matrix-class object, that could be of class \code{gpu.matrix.torch} or \code{gpu.matrix.tensorflow} depending on the system installed in the computer.

The matrix created will be stored in the GPU (by default) or in the CPU. The example section explains how to be sure where the matrix is stored.

This function also mimics the function \code{Matrix} of the library 'Matrix'.
}
\usage{
gpu.matrix(data = NULL, nrow = NULL, ncol = NULL,
           byrow = FALSE,dimnames = NULL,
           dtype=NULL, sparse=NULL,colnames=c(),
           rownames=c(),device=NULL,type=NULL)

as.gpu.matrix(x,\dots)
\S4method{as.gpu.matrix}{ANY}(x,\dots)
}
\arguments{
  \item{data,x}{
a scalar, vector or matrix (both matrix or Matrix class).
}
  \item{nrow}{
Number of rows of the matrix. By default the number of rows of data if data is an object of class matrix or Matrix.
}
  \item{ncol}{
Number of columns of the matrix. By default the number of columns of data if data is an object of class matrix or Matrix.
}
  \item{byrow}{
The same as  function \code{matrix}: "logical. If FALSE (the default) the matrix is filled by columns, otherwise the matrix is filled by rows."
}
  \item{dimnames}{
The same as in  function \code{matrix}: "A dimnames attribute for the matrix: NULL or a list of length 2 giving the row and column names respectively. An empty list is treated as NULL, and a list of length one as row names. The list can be named, and the list names will be used as names for the dimensions."
}
  \item{dtype}{
data type. User can indicate "float64", "float32" or "int" for "int64". if not specified, dtype will correspond to the input data type.
}
  \item{sparse}{
The same as in function \code{Matrix} of the library 'Matrix': "logical or NULL, specifying if the result should be sparse or not. By default, it is made sparse when more than half of the entries are 0."
}
  \item{colnames}{
A vector with the column names.
}
  \item{rownames}{
A vector with the row names.
}
\item{type}{If the gpu.matrix is 'torch' or "tensorflow". If it is NULL, \code{gpu.matrix} will try to create a gpu.matrix.torch object.}
\item{device}{It indicates the device to load cuda. If not indicated, 'device' will be set to 'cuda' if it is available.}
\item{\dots}{additional arguments to be passed to or from methods.}

}
\details{
The \code{gpu.matrix} function mimics the \code{Matrix} function of the 'Matrix' library and the basic \code{matrix} function. If tensorflow and/or torch are properly installed and the \code{device} parameter is set to "cuda" (by default), then the created gpu.matrix object will be stored on the GPU. The example shows how to check this.

The user can apply to the created gpu.matrix-class object -using the same operators- the basic functions that can be applied to a object of class 'matrix' and/or class 'Matrix'.

It can also work with sparse matrices as the 'Matrix' library.
}
\value{
Returns a GPUmatrix object that can be either "gpu.matrix.tensorflow" or "gpu.matrix.torch". For both torch and tensorflow the functions to be applied to a matrix are the same.

If the gpu.matrix-class object is not sparse it will show on the console the matrix as it is. If the gpu.matrix is sparse, it will return to the console the position where there are number different from zero. The internal values of the matrix can be seen using the operator "@".

If the gpu.matrix-class object contains complex numbers, to access the real and imaginary information use the function \code{Re()} for teh rea part and \code{Im()} for the imaginary part. Furthermore, the following code can be used:
\code{output@gm$real} for the real part and \code{output@gm$imag} for the imaginary part.

Even if the gpu.matrix-class object is sparse or not, both kind of matrices works equally with all functions.
}
\author{
Cesar Lobato and Angel Rubio.
}
\seealso{
See \code{\linkS4class{gpu.matrix}}, \code{\linkS4class{Matrix}}, and \code{\linkS4class{matrix}}.

For more details about the parameter \code{dtype} visit \code{\link{dtype}}
}
\examples{

\donttest{
\dontrun{
## create a gpu.matrix.torch and check it is stored in the GPU.
a <- gpu.matrix(1:9,nrow=3,ncol=3)
class(a)
a@gm$is_cuda

# the output of class(a) should be:
#[1] "gpu.matrix.torch"
#attr(,"package")
#[1] "GPUmatrix"

#the output of a@gm@device should have a similar shape:
#[1] TRUE

## create a gpu.matrix.torch and check it is stored in the CPU.
a <- gpu.matrix(1:9,nrow=3,ncol=3, device="cpu")
class(a)
a@gm$is_cuda

# the output of class(a) should be:
#[1] "gpu.matrix.torch"
#attr(,"package")
#[1] "GPUmatrix"

#the output of a@gm@device should have a similar shape:
#[1] FALSE

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

#create a sparse
a <- gpu.matrix(data=c(0,1,1,0,1,0),nrow = 3,ncol = 2,sparse = T)
a

#create a complex gpu.matrix
a <- gpu.matrix(data=c(0+1i,1i,1,0,1,0),nrow = 3,ncol = 2)
a




}
}
}

