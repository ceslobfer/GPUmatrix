\name{apply}
\title{Apply Functions over 'gpu.matrix-class' margins}

\description{

This function mimics the 'base' function \code{'apply'} to operate on gpu.matrix-class objects: It returns a vector or a list of values obtained by applying a function to margins of a GPUmatrix.

}
\alias{apply}
\alias{apply-methods}
\alias{apply,gpu.matrix.tensorflow-method}
\alias{apply,gpu.matrix.torch-method}

\usage{
\S4method{apply}{gpu.matrix.tensorflow}(X, MARGIN, FUN, \dots, simplify)
\S4method{apply}{gpu.matrix.torch}(X, MARGIN, FUN, \dots, simplify)
}

\arguments{
\item{X}{a \code{\linkS4class{gpu.matrix}} object.}
\item{MARGIN}{1 for rows and 2 for columns.}
\item{FUN}{function to be applied in the operation.}
\item{\dots}{general additional parameters. Optional arguments to FUN.}
\item{simplify}{a logical indicating whether results should be simplified if possible. Note that some methods that can be simplified when working with 'matrix' objects may not always be simplified for gpu.matrix objects. See details.}
}

\details{
\code{FUN} is found by a call to \code{\link[base]{match.fun}} as done in the base function \code{\link[base]{apply}}. Internally, apply will use the functions implemented to work with objects from the GPUmatrix library. If the input gpu.matrix-class object(s) are stored on the GPU, then the operations will be performed on the GPU. See \code{\link{gpu.matrix}}.

As in \code{\link[base]{apply}}, the arguments in \code{...} cannot have the same name as any of the other arguments to ensure possible errors.

The parameter \code{simplify} indicates wheter the result should be simplified if possible. If the called \code{FUN} returns a gpu.matrix-class object, the result cannot be simplified. In these cases, the parameter \code{simplify} will work as if it was set to \code{FALSE} and the following warning message will be returned: "If the function applied to the GPU matrix returns a tensor or another GPU matrix, then the 'simplify' argument will always be FALSE."}


\value{
The results of mimics the base function \code{\link[base]{apply}}.

Each call to \code{FUN} will return a vector of length n. If \code{simplify} is TRUE and the result can be simplified, then apply will return a numeric vector of dimension \code{c(n,dim(x)[MARGIN])} if \code{n > 1}. If \code{n = 1}, \code{apply} will return a numeric vector of length \code{dim(x)[MARGIN]}.

If simplify is FALSE, apply will return a list of length \code{dim(x)[MARGIN]}.

Note that if \code{simplify} is TRUE and the result of \code{FUN} is an object of class gpu.matrix, then the result cannot be simplified, so it will return a list of length \code{dim(x)[MARGIN]} and each element of this list will be of class gpu.matrix.

For more details see \code{\link[base]{apply}}
}


\seealso{
For more information see:
\code{\link[base]{apply}}
}

\examples{

if(installTorch()){

  a <- gpu.matrix(rnorm(9),3,3)

  apply(a, 1, mean) #computes the mean of each row
  apply(a, 2, mean) #computes the mean of each column

}




}












