\name{matrix_general_operators_methods}
\title{Return the first or last part of a GPUmatrix object}
\docType{methods}

\description{
\code{head} and \code{tail} mimic the functions \code{head} and \code{tail} from \code{utils} to operate on gpu.matrix-class objects. By default \code{head} shows the first 6 rows of a matrix or first 6 elements of a vector or list. \code{tail} shows the last 6 rows of a matrix or last 6 elements of a vector or list.

The function \code{show} mimics the function \code{show} of \code{methods} to compute on gpu.matrix-class objects: "It display the object, by printing, plotting or whatever suits its class."

The function \code{print} mimics the base function \code{print} to operate on gpu.matrix-class objects.

}


\alias{tail}
\alias{show}
\alias{head}

\alias{tail-methods}
\alias{show-methods}
\alias{head-methods}
\alias{tail,gpu.matrix.tensorflow-method}
\alias{tail,gpu.matrix.torch-method}
\alias{show,gpu.matrix.tensorflow-method}
\alias{show,gpu.matrix.torch-method}
\alias{head,gpu.matrix.tensorflow-method}
\alias{head,gpu.matrix.torch-method}

\alias{print}
\alias{print-methods}
\alias{print,gpu.matrix.torch-method}


\usage{

\S4method{tail}{gpu.matrix.tensorflow}(x,\dots)
\S4method{tail}{gpu.matrix.torch}(x,\dots)
\S4method{show}{gpu.matrix.tensorflow}(object)
\S4method{show}{gpu.matrix.torch}(object)
\S4method{head}{gpu.matrix.tensorflow}(x,\dots)
\S4method{head}{gpu.matrix.torch}(x,\dots)

\S4method{print}{gpu.matrix.torch}(x)

}


\arguments{
\item{x,object}{a \code{\linkS4class{gpu.matrix}}.}
\item{\dots}{arguments to be passed to or from other methods.}
}

\seealso{
For more information see:
\code{\link[utils]{head}}, \code{\link[utils]{tail}}, and \code{\link[methods]{show}}.
}
\examples{
\donttest{
\dontrun{
a <- gpu.matrix(rnorm(20*5),20,5)
head(a) #shows the first six row of every column
tail(a) #shows the las six row of every column

show(a) #show all the object
a #equivalente to run the function show.
}
}
}


