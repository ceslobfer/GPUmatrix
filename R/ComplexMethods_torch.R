setMethod("Complex",
          c(z="gpu.matrix.torch"),
          function(z)
          {
            op = .Generic[[1]]
            switch(op,
                   'Re' = {
                     z@gm <- z@gm$real
                   },
                   'Im' = {
                     z@gm <- z@gm$imag
                   },
                   'Conj' = {
                     z@gm <- z@gm$conj()
                   },
                   'Arg' = {
                     z@gm <- z@gm$angle()
                   },
                   'Mod'={
                     z@gm <- z@gm$abs()
                   }
            )
            return(z)
          }
)
