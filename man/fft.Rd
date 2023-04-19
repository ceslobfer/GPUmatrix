\name{fft}
\title{Fast Discrete Fourier Transform (FFT)}


\description{

Computes the fft of the rows of a matrix. It is equivalent to the base R  \code{apply(A,1,fft)}.

}

\alias{fft}
\alias{fft-methods}
\alias{fft,gpu.matrix.tensorflow-method}
\alias{fft,gpu.matrix.torch-method}


\usage{
\S4method{fft}{gpu.matrix.tensorflow}(z)
\S4method{fft}{gpu.matrix.torch}(z)
}

\arguments{
\item{z}{a real or complex gpu.matrix containing the values to be transformed.}
}

\seealso{
For more information see:
\code{\link[stats]{fft}}
}

\value{
It returns the corresponding output of the function 'fft' of the library 'stats':  the unnormalized univariate discrete Fourier transform of the sequence of values in input \code{z}. For more details check the corresponding help of 'fft' of the library 'stats'.
}


\examples{
if(installTorch){

  x <- gpu.matrix(1:4)
  fft(x)

}


}



