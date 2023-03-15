\name{expmGPU}
\alias{expmGPU}
\alias{expmGPU-methods}
\alias{expmGPU,gpu.matrix.tensorflow-method}
\title{
GPUMatrix Exponential.
}
\description{
It is a mimic of the function \code{expm} of the library \code{Matrix}. It computes the exponential of a GPUmatrix.
}
\usage{
expmGPU(x)
\S4method{expmGPU}{gpu.matrix.tensorflow}(x)
}
\arguments{
  \item{x}{a gpu.matrix build with float numbers}
}
\value{
the matrix exponential of x as GPUmatrix class.
}
\details{
This functios works with float numbers. If the numbers used to create the gpu.matrix are integers, this function will not work. An example is shown below.
}

\seealso{
See Also as \code{\link[Matrix]{expm}}.
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

