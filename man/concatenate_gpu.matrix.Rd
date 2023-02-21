\name{concatenate_gpu.matrix}
\title{concatenate_gpu.matrix}
\docType{methods}



\alias{c-methods}

\alias{c,gpu.matrix.tensorflow-method}

\alias{c,gpu.matrix.torch-method}

\alias{c,numMatrixLike-method}




\description{
Mimic of c basic function: This is a generic
function which combines its arguments.
}


\usage{

\S4method{c}{gpu.matrix.tensorflow}(x,\dots,recursive) 

\S4method{c}{gpu.matrix.torch}(x,\dots,recursive) 

\S4method{c}{numMatrixLike}(x,\dots,recursive) 

}


\arguments{

\item{x}{A gpu.matrix.tensorflow o gpu.matrix.torch object}

\item{\dots}{objects to be concatenated.}

\item{recursive}{logical. If recursive = TRUE, the function recursively descends
                          through lists (and pairlists) combining all their elements into a vector.}

}
