setMethod("Complex",
          c(z="gpu.matrix.tensorflow"),
          function(z)
          {
            op = .Generic[[1]]
            switch(op,
                   'Re' = {
                     z@gm <- tf$math$real(z@gm)
                   },
                   'Im' = {
                     z@gm <- tf$math$imag(z@gm)
                   },
                   'Conj' = {
                     z@gm <- tf$math$conj(z@gm)
                   },
                   'Arg' = {
                     z@gm <- tf$math$angle(z@gm)
                   },
                   'Mod'={
                     z@gm <- tf$math$abs(z@gm)
                   }
            )
            return(z)
          }
)
