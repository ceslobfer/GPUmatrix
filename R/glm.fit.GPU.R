# glm.fit.GPU <- function (x,y, intercept = TRUE, weights = NULL, row.chunk = NULL,
#                          family = gaussian(), start = NULL, etastart = NULL, mustart = NULL,
#                          offset = NULL, acc = 1e-08, maxit = 25, k = 2, sparselim = 0.9,
#                          camp = 0.01, eigendec = FALSE, tol.values = 1e-07, tol.vectors = 1e-07,
#                          tol.solve = .Machine$double.eps, sparse = NULL, method = c("eigen",
#                                                                                     "Cholesky", "qr"), trace = FALSE, ...)
# {
#   ## TODO: remove the code related to methods eigen, Cholesky, qr. Later on we can add more methods.
#   ## eigendec = TRUE? What is that for? Control does not work with it.
#   ## Add sparse argument if the matrix is desired to be treated as sparse. Maybe x could be a sparse Matrix argument
#   ## Add argument to select tensorflow or torch
#   ## Add argument to select cpu or gpu
#   ## Add argument to select float32 or float64
#   ## Implement pmax so that no need to cast in the linkinv function... dgamma is a tough one.
#
#
#   nobs <- NROW(y)
#   nvar <- ncol(x)
#
#   #Initial checks
#   if (missing(y))
#     stop("Argument y is missing")
#   if (missing(x))
#     stop("Argument x is missing")
#   if (is.null(offset))
#     offset <- rep.int(0, nobs)
#   if (is.null(weights))
#     weights <- rep(1, nobs)
#
#   #Sets conditions
#   col.names <- dimnames(x)[[2]]
#   method <- match.arg(method)
#   fam <- family$family
#   link <- family$link
#   variance <- family$variance
#   dev.resids <- family$dev.resids
#   aic <- family$aic
#   linkinv <- family$linkinv
#   mu.eta <- family$mu.eta
#
#   #Spaghettifelse
#   if (is.null(sparse))
#     sparse <- F ####Fix me!
#   if (is.null(start)) {
#     if (is.null(mustart))
#       eval(family$initialize)
#     eta <- if (is.null(etastart))
#       family$linkfun(mustart)
#     else etastart
#     mu <- mustart
#     start <- rep(0, nvar)
#   }
#   else {
#     eta <- offset + as.vector(if (nvar == 1)
#       x * start
#       else {
#         if (sparse)
#           x %*% start
#         else tcrossprod(x, t(start))
#       })
#     mu <- linkinv(eta)
#   }
#   iter <- 0
#   dev <- sum(dev.resids(y, c(mu), weights))
#
#   #GPUmatrix initialization
#   x <- gpu.matrix(x)
#   y <- gpu.matrix(y)
#   tol <- 1
#
#   if ((fam == "gaussian") & (link == "identity"))
#     maxit <- 1
#
#   while ((tol > acc) & (iter < maxit)) {
#     iter <- iter + 1
#     beta <- start
#     dev0 <- dev
#     varmu <- variance(mu)
#     mu.eta.val <- mu.eta(as.numeric(eta))
#     z <- (eta - offset) + (y - mu)/mu.eta.val
#     W <- gpu.matrix((weights * mu.eta.val * mu.eta.val)/varmu)
#     XTX <- crossprod(x*W, x)
#     XTz <- t(crossprod((W * z), x))
#     if (iter == 1 & method != "qr") {
#       variable <- colnames(x)
#       ris <- if (eigendec)
#         control(XTX, , tol.values, tol.vectors, , method)
#       else list(rank = nvar, pivot = 1:nvar)
#       ok <- ris$pivot[1:ris$rank]
#       if (eigendec) {
#         XTX <- ris$XTX
#         x <- x[, ok]
#         XTz <- XTz[ok]
#         start <- start[ok]
#       }
#       beta <- start
#     }
#     if (method == "qr") {
#       ris <- qr(XTX, tol.values)
#       ris$coefficients <- drop(qr.solve(ris, XTz, tol.values))
#       start <- if (ris$rank < nvar)
#         ris$coefficients[ris$pivot]
#       else ris$coefficients
#     }
#     else {
#       start <- solve(XTX, XTz)
#     }
#     eta <- if (sparse)
#       drop(x %*% start)
#     else drop(tcrossprod(x, t(start)))
#     mu <- linkinv(as.numeric(eta <- eta + offset))
#     dev <- sum(dev.resids(as.numeric(y), c(mu), weights))
#     tol <- max(abs(dev0 - dev)/(abs(dev) + 0.1))
#     if (trace)
#       cat("iter", iter, "tol", tol, "\n")
#   }
#
#   wt <- sum(weights)
#   wtdmu <- if (intercept)
#     as.numeric(sum(weights * y)/wt)
#   else linkinv(offset)
#   nulldev <- sum(dev.resids(as.numeric(y), c(wtdmu), weights))
#   n.ok <- nobs - sum(weights == 0)
#   nulldf <- n.ok - as.integer(intercept)
#   rank <- ris$rank
#   dfr <- nobs - rank - sum(weights == 0)
#   aic.model <- aic(as.numeric(y), nobs, as.numeric(mu), weights, dev) + k * rank
#   ll.nuovo <- ll.speedglmGPU(fam, aic.model, rank) # FIX ME!! --> Remove dependency from speedglm
#   res <- (y - mu)/mu.eta(as.numeric(eta))
#   resdf <- n.ok - rank
#   RSS <- sum(W * res * res)
#   var_res <- RSS/dfr
#   dispersion <- if (fam %in% c("poisson", "binomial"))
#     1
#   else var_res
#   if (method == "qr") {
#     coefficients <- start
#     coefficients[coefficients == 0] = NA
#     ok <- ris$pivot[1:rank]
#   }
#   else {
#     coefficients <- rep(NA, nvar)
#     start <- as(start, "numeric")
#     coefficients[ok] <- start
#   }
#   names(coefficients) <- if (is.null(col.names) & (!is.null(coefficients))) {
#     if (intercept) {
#       if (length(coefficients) > 1)
#         c("(Intercept)", paste("V", 1:(length(coefficients) -
#                                          1), sep = ""))
#       else "(Intercept)"
#     }
#     else paste("V", 1:length(coefficients), sep = "")
#   }
#   else col.names
#   rval <- list(coefficients = coefficients, logLik = ll.nuovo,
#                iter = iter, tol = tol, family = family, link = link,
#                df = dfr, XTX = as.matrix(XTX), dispersion = dispersion, ok = ok,
#                rank = rank, RSS = as.numeric(RSS), method = method, aic = aic.model,
#                offset = offset, sparse = sparse, deviance = dev, nulldf = nulldf,
#                nulldev = nulldev, ngoodobs = n.ok, n = nobs, intercept = intercept,
#                convergence = (!(tol > acc)))
#   class(rval) <- "speedglm"
#   rval
# }
#
# ll.speedglmGPU <- function (family, aic.model, nvar) {
#   switch(family,
#          binomial = -(aic.model - 2 * nvar)/2,
#          Gamma = -((aic.model - 2 * nvar) - 2)/2,
#          gaussian = -((aic.model - 2 * nvar) - 2)/2,
#          poisson = -(aic.model - 2 * nvar)/2,
#          inverse.gaussian = -((aic.model - 2 * nvar) - 2)/2,
#          quasi = NA,
#          quasibinomial = NA,
#          quasipoisson = NA)
# }
