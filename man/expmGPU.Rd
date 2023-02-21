\name{expmGPU}
\alias{expmGPU}
\title{
Matrix Exponential with the GPU.
}
\description{
It is a mimic of the function 'expm' of the library 'Matrix'. It computes the exponential of a matrix using the GPU.
}
\usage{
expmGPU(x)
}
\arguments{
  \item{x}{a gpu.matrix build with float numbers}
}
\details{
For more details see the 'expm' function of the library 'Matrix'.
}
\value{
the matrix exponential of x.
}
\references{
https://en.wikipedia.org/wiki/Matrix_exponential

Cleve Moler and Charles Van Loan (2003) Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later. SIAM Review 45, 1, 3–49.

Eric W. Weisstein et al. (1999) Matrix Exponential. From MathWorld, https://mathworld.wolfram.com/MatrixExponential.html
}
\note{
This functios works with float numbers. If the numbers used to create the gpu.matrix are integers, this function will not work. An example is shown below.
}

\seealso{
See Also as \code{\link{expm}}
}
\examples{
  \donttest{
  \dontrun{
  #this first example is build with a matrix that contains int number. It will  not work.
  x <- gpu.matrix(1:9,nrow=3,ncol = 3)
  try(expmGPU(x))
  #need to be float and not int32

  #To work with float, we can add as.numeric. The following code will work:
  x <- gpu.matrix(as.numeric(1:9),nrow=3,ncol = 3)
  try(expmGPU(x))
}
}

}

