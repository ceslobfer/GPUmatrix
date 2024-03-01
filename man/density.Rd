\name{density}
\title{Kernel Density Estimation and Histograms}
\description{
The function \code{density} mimics the function \code{density} of the library \code{stats} to operate on gpu.matrix-class objects: "It computes kernel density estimates. Its default method does so with the given kernel and bandwidth for univariate observations."

The function \code{'hist'} mimics the function \code{'hist'} of the library \code{'graphics'} to operate on gpu.matrix-class objects: "It computes a histogram of the given data values."

}

\alias{density}
\alias{density-methods}
\alias{density,gpu.matrix.tensorflow-method}
\alias{density,gpu.matrix.torch-method}

\alias{hist}
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
\item{x}{the \code{\linkS4class{gpu.matrix}} object from which the estimate density is to be computed or the histogram is desired.}
\item{\dots}{further arguments and graphical parameters.}
}

\seealso{
For more information see:
\code{\link[stats]{density}}, and
\code{\link[graphics]{hist}}

}

\details{
The two functions (\code{density} and \code{hist}) have been programmed to call their corresponding counterpart functions with their default parameters. Therefore, the internal operations to obtain each graph are computed by the CPU, regardless of whether the input value is stored in the GPU.

For more information on these functions see \code{\link[stats]{density}}, and \code{\link[graphics]{hist}}.

}

\value{

The function \code{density} returns the same output as its counterpart function \code{density} from the library \code{stats}: It returns "an object with class 'density' whose underlying structure is a list containing the following components.

\item{x}{the n coordinates of the points where the density is estimated.}
\item{y}{the estimated density values. These will be non-negative, but can be zero.}
\item{bw}{the bandwidth used.}
\item{n}{the sample size after elimination of missing values.}
\item{call}{the call which produced the result.}
\item{data.name}{	the deparsed name of the x argument.}
\item{has.na}{logical, for compatibility (always FALSE).}

The print method reports summary values on the x and y components." (taken from \code{\link[stats]{density}}).

On the other hand, the function \code{hist} returns the same output as its counterpart function \code{hist} from the library \code{graphics}: It returns "an object of class 'histogram' which is a list with components:

\item{breaks}{the n+1n+1 cell boundaries (= breaks if that was a vector). These are the nominal breaks, not with the boundary fuzz.}
\item{counts}{n integers; for each cell, the number of x[] inside.}
\item{density}{values \eqn{\hat{f}(x_i)}, as estimated density values. If \code{all(diff(breaks) == 1)}, the are the relative frequencies \code{counts/n} and in general satisfy \eqn{\sum_i{\hat{f}(x_i)(b_{i+1}-b_i)=1}}, where \eqn{b_i} = \code{breaks[i]}}.
\item{mids}{the n cell midpoints.}
\item{xname}{a character string with the actual x argument name.}
\item{equidist}{logical, indicating if the distances between breaks are all the same."}
(Taken from \code{\link[graphics]{hist}})
}

\examples{

if(installTorch()){

  a <- gpu.matrix(rnorm(20*100),20,100)

  density(a[1,]) #density information
  plot(density(a[1,])) #plot the estimated density function

  hist(a[1,]) #plot the histogram

}


}




