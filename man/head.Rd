\name{matrix_general_operators_methods}
\title{matrix_general_operators_methods}
\docType{methods}

\description{

\code{head} and \code{tail} mimic the functions \code{head} and \code{tail} from \code{utils}. They return the first or last parts of a GPUmatrix object.

The function \code{show} mimics the function \code{show} of \code{methods}. It Display the object, by printing, plotting or whatever suits its class.

}


\alias{tail-methods}
\alias{show-methods}
\alias{head-methods}
\alias{tail,gpu.matrix.tensorflow-method}
\alias{tail,gpu.matrix.torch-method}
\alias{show,gpu.matrix.tensorflow-method}
\alias{show,gpu.matrix.torch-method}
\alias{head,gpu.matrix.tensorflow-method}
\alias{head,gpu.matrix.torch-method}



\usage{

\S4method{tail}{gpu.matrix.tensorflow}(x,\dots)
\S4method{tail}{gpu.matrix.torch}(x,\dots)
\S4method{show}{gpu.matrix.tensorflow}(object)
\S4method{show}{gpu.matrix.torch}(object)
\S4method{head}{gpu.matrix.tensorflow}(x,\dots)
\S4method{head}{gpu.matrix.torch}(x,\dots)

}


\arguments{

\item{x,object}{a GPUmatrix object}
\item{\dots}{arguments to be passed to or from other methods.}

}

\seealso{
For more information see:
\code{\link[utils]{head}}, \code{\link[utils]{tail}}, and \code{\link[methods]{show}}.
}



\examples{
\donttest{
\dontrun{

a <- gpu.matrix(rnorm(20*5),20,5,type="tensorflow")

head(a) #shows the first six row of every column
tail(a) #shows the las six row of every column

show(a) #show all the object
a #equivalente to apply function show.
}
}
}


