\name{density}
\title{Kernel Density Estimation}


\description{

The function \code{density} mimics the function \code{density} of the library \code{stats}. It computes kernel density estimates. Its default method does so with the given kernel and bandwidth for univariate observations.

The function \code{hist} mimics the function \code{hist} of the library \code{graphics}. It computes a histogram of the given data values.

}

\alias{density-methods}
\alias{density,gpu.matrix.tensorflow-method}
\alias{density,gpu.matrix.torch-method}

\alias{hist-methods}
\alias{hist,gpu.matrix.tensorflow-method}
\alias{hist,gpu.matrix.torch-method}


\usage{
\S4method{density}{gpu.matrix.tensorflow}(x)
\S4method{density}{gpu.matrix.torch}(x)
\S4method{hist}{gpu.matrix.tensorflow}(x,\dots)
\S4method{hist}{gpu.matrix.torch}(x,\dots)

}


\arguments{
\item{x}{the GPUmatrix object from which the estimate density is to be computed or the GPUmatrix object for which the histogram is desired.}
\item{\dots}{further arguments and graphical parameters.}
}

\seealso{
For more information see:
\code{\link[stats]{density}}, and
\code{\link[graphics]{hist}}

}

\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(20*100),20,100,type="tensorflow")

density(a[1,]) #density information
plot(density(a[1,])) #plot the estimated density function

hist(a[1,]) #plot the histogram

}
}
}




