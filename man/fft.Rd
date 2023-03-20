\name{fft}
\title{Fast Discrete Fourier Transform (FFT)}


\description{

It mimics the function \code{fft} of library \code{stats}. It computes the Discrete Fourier Transform (DFT) of an array with a fast algorithm, the “Fast Fourier Transform” (FFT).

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


\examples{
\donttest{
\dontrun{

x <- gpu.matrix(1:4)
fft(x)



}
}
}



