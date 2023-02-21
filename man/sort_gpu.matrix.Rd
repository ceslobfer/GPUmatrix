\name{sort_gpu.matrix}
\title{sort_gpu.matrix}
\docType{methods}



\alias{sort-methods}

\alias{sort,gpu.matrix.tensorflow,logical-method}

\alias{sort,gpu.matrix.tensorflow,missing-method}

\alias{sort,gpu.matrix.torch,logical-method}

\alias{sort,gpu.matrix.torch,missing-method}




\description{
Mimic of sort base function: Sort (or order) a 
a gpu.matrix object (partially) into ascending or descending order.
}


\usage{

\S4method{sort}{gpu.matrix.tensorflow,logical}(x,decreasing) 

\S4method{sort}{gpu.matrix.tensorflow,missing}(x,decreasing) 

\S4method{sort}{gpu.matrix.torch,logical}(x,decreasing) 

\S4method{sort}{gpu.matrix.torch,missing}(x,decreasing) 

}


\arguments{

\item{x}{A gpu.matrix.torch or a gpu.matrix.tensorflow object.}

\item{decreasing}{Logical. Should the sort be increasing or decreasing? }

}
