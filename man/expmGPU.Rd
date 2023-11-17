\name{expmGPU}
\alias{expmGPU}
\alias{expmGPU-methods}
\alias{expmGPU,gpu.matrix.tensorflow-method}
\alias{expmGPU,gpu.matrix.torch-method}
\title{
'GPUmatrix' Exponential.
}
\description{
It mimics the function \code{expm} of the library \code{Matrix} to operate on gpu.matrix-class objects: It "computes the exponential of a matrix."
}
\usage{
expmGPU(x)
\S4method{expmGPU}{gpu.matrix.tensorflow}(x)
\S4method{expmGPU}{gpu.matrix.torch}(x)
}
\arguments{
  \item{x}{a \code{\linkS4class{gpu.matrix}}.}
}
\value{
The matrix exponential of \code{x} as \code{\linkS4class{gpu.matrix}} class.
}
\details{

The exponential of a matrix is computed as:
\eqn{\sum_{k=0}^\infin 1/k!X^k}

The function \code{expmGPU} mimics the function \code{expm} of the library \code{Matrix} by applying the Ward's diagonal Pade' approximation. For more details see \code{\link[Matrix]{expm}}.

Please note that this function works with float numbers (either float32 or float64). If the data type of \code{x} is integer, this function will not work. An example is shown below.
}

\seealso{
For more information see  \code{\link[Matrix]{expm}}.
}
\examples{
  \donttest{
  \dontrun{
  #build with a matrix that contains int number. It will  not work.
x <- gpu.matrix(1:9,nrow=3,ncol = 3,dtype = "int")
x
try(expmGPU(x))
#need to be float and not int32

#by default, gpu.matrix work with float64
x <- gpu.matrix(1:9,nrow=3,ncol = 3)
try(expmGPU(x))
}
}

}

