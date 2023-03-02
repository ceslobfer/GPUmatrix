setMethod("Math",
          signature(x="gpu.matrix.torch"),
          function(x)
          {
            op = .Generic[[1]]

            switch(op,
                   'log' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$log()
                     return(x)
                   },
                   'log2' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$log2()
                     return(x)
                   },
                   'log10' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$log10()
                     return(x)
                   },
                   'log1p' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$log1p()
                     return(x)
                   },
                   'cos' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$cos()
                     return(x)
                   },
                   'cosh' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$cosh()
                     return(x)
                   },
                   'acos' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$acos()
                     return(x)
                   },
                   'acosh' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$acosh()
                     return(x)
                   },
                   'sin' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$sin()
                     return(x)
                   },
                   'sinh' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$sinh()
                     return(x)
                   },
                   'asin' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$asin()
                     return(x)
                   },
                   'asinh' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$asinh()
                     return(x)
                   },
                   'tan' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$tan()
                     return(x)
                   },
                   'atan' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$atan()
                     return(x)
                   },
                   'tanh' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$tanh()
                     return(x)
                   },
                   'atanh' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$atanh()
                     return(x)
                   },
                   'sqrt' = {
                     x@gm <- x@gm$sqrt()
                     return(x)
                   },
                   'abs' = {
                     x@gm <- x@gm$abs()
                     return(x)
                   },
                   'sign' = {
                     x@gm <- x@gm$sign()
                     return(x)
                   },
                   'ceiling' = {
                     x <- warningInteger(x)
                     x@gm <- x@gm$ceil()
                     return(x)
                   },
                   'floor' = {
                     x <- warningInteger(x)
                     x@gm <- x@gm$floor()
                     return(x)
                   },
                   'cumsum' = {
                     if (x@sparse) {
                       res <- gpu.matrix(x@gm$values()$cumsum(1,))
                     }else{
                       dim(x) <- c(1,length(x))
                       res <- x@gm$cumsum(2,)
                     }

                     return(gpu.matrix(res))
                   },
                   'cumprod' = {
                     if (x@sparse) {
                       res <- gpu.matrix(x@gm$values()$cumprod(1,))
                     }else{
                       dim(x) <- c(1,length(x))
                       res <- x@gm$cumprod(2,)
                     }

                     return(gpu.matrix(res))
                   },
                   'exp' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$exp()
                     return(x)
                   },
                   'expm1' = {
                     x <- warningSparseTensor_torch(x)
                     x@gm <- x@gm$expm1()
                     return(x)
                   }
            )
          }
)
