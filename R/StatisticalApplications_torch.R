
updateH <- function(V,W,H) {
  H <- H * (t(W) %*% V)/((t(W) %*% W) %*% H)
}

updateW <- function(V,W,H) {
  W <- W * (V %*% t(H))/(W %*% (H %*% t(H)) )
}


NMFgpumatrix <- function(V,k=10,Winit=NULL, Hinit=NULL, tol=1e-6, niter=100){
  set.seed(123)
  if (class(V)[[1]] == "gpu.matrix.torch") {
    if(is.null(Winit)) Winit <- gpu.matrix(runif(nrow(V)*k),nrow(V),k, dtype = dtype(V),device = GPUmatrix:::device(V))
    if(is.null(Hinit)) Hinit <- gpu.matrix(runif(k*ncol(V)),k,ncol(V), dtype = dtype(V),device = GPUmatrix:::device(V))
  }else{
    if(is.null(Winit)) Winit <- matrix(runif(nrow(V)*k),nrow(V),k)
    if(is.null(Hinit)) Hinit <- matrix(runif(k*ncol(V)),k,ncol(V))
  }

  Vold <- V
  condition <- F
  for (iter in 1:niter) {
    Winit <- updateW(V,Winit,Hinit)
    Hinit <- updateH(V,Winit,Hinit)
    Vnew <- Winit%*%Hinit
    if(mean((Vnew-Vold)^2)<tol){
      res <- list("W"=Winit,"H"=Hinit)
      condition <- T
      break()
    }
    Vold <- Vnew
  }

  if(!condition){
    warning(message="Early finish")
  }
  return(res)
}



# Define the logistic function
sigmoid <- function(x) {
  1/(1+exp(-x))
}
# Defin the function to train a logistic regression
# using the conjugate gradient
LR_GradientConjugate_gpumatrix <- function(X,y,beta = NULL, lambda = 0, iterations = 1000, tol = 1e-6) {
  tX <- t(X)
  if (is.null(beta))
    beta <- solve(crossprod(X),crossprod(X,2*y-1)) # Returns double even with float inputs. Fix!
  p <- sigmoid(X %*% beta)
  a <- p*(1-p)
  g <- tX %*% (p - y)
  u <- g
  # u_old <- u
  for (iter in 1:iterations) {
    if (iter == 1) {
      uhu <- (sum(u*(tX %*% (a * (X %*% u)))) + lambda * sum(u*u))
      beta <- beta-sum(g*u)/uhu * u
    } else{
      p <- sigmoid(X %*% beta)
      # beta_old <- beta
      g_old <- g
      a <- p*(1-p)
      g <- tX %*% (p - y)
      k <- g - g_old
      beta_coef <- sum(g * k)/sum(u*k) # Hestenes-Stiefel update. Other options are possible
      u <- g - u * beta_coef
      # u_old <- u
      uhu <- sum((X %*% u)^2*a) + lambda * sum(u*u)
      beta <- beta - sum(g*u)/uhu * u
      if(sum(g*g)< tol)
        break
    }
  }
  return(beta)
}


GPUglmfit <- function(...) {
  res <- glm(..., method = "glm.fit.GPU")
  # class(res) <- c("GPUglm")
  res <- new("GPUglm", glm=res)
  return(res)
}


glm.fit.GPU <- function (x,y, intercept = TRUE, weights = NULL, row.chunk = NULL,
                         family = gaussian(), start = NULL, etastart = NULL, mustart = NULL,
                         offset = NULL, acc = 1e-08, maxit = 25, k = 2, sparselim = 0.9,
                         camp = 0.01, eigendec = FALSE, tol.values = 1e-07, tol.vectors = 1e-07,
                         tol.solve = .Machine$double.eps, sparse = NULL,
                         method = c("eigen","Cholesky", "qr"), trace = FALSE,
                         dtype=NULL,
                         device=NULL,
                         type=NULL,...)
{
  ## TODO: remove the code related to methods eigen, Cholesky, qr. Later on we can add more methods. CHECK
  ## eigendec = TRUE? What is that for? Control does not work with it. CHECK
  ## Add sparse argument if the matrix is desired to be treated as sparse. Maybe x could be a sparse Matrix argument
  ## Add argument to select tensorflow or torch
  ## Add argument to select cpu or gpu
  ## Add argument to select float32 or float64
  ## Implement pmax so that no need to cast in the linkinv function... dgamma is a tough one.


  nobs <- NROW(y)
  nvar <- ncol(x)

  #Initial checks
  if (missing(y))
    stop("Argument y is missing")
  if (missing(x))
    stop("Argument x is missing")
  if (is.null(offset))
    offset <- rep.int(0, nobs)
  if (is.null(weights))
    weights <- rep(1, nobs)

  #Sets conditions
  col.names <- dimnames(x)[[2]]
  method <- match.arg(method)
  fam <- family$family
  link <- family$link
  variance <- family$variance
  dev.resids <- family$dev.resids
  aic <- family$aic
  linkinv <- family$linkinv
  mu.eta <- family$mu.eta

  #Spaghettifelse
  if (is.null(sparse))
    sparse <- F ####Fix me!
  if (is.null(start)) {
    if (is.null(mustart))
      eval(family$initialize)
    eta <- if (is.null(etastart))
      family$linkfun(mustart)
    else etastart
    mu <- mustart
    start <- rep(0, nvar)
  }
  else {
    eta <- offset + as.vector(if (nvar == 1)
      x * start
      else {
        if (sparse)
          x %*% start
        else tcrossprod(x, t(start))
      })
    mu <- linkinv(eta)
  }
  iter <- 0
  dev <- sum(dev.resids(y, c(mu), weights))

  #GPUmatrix initialization
  x <- gpu.matrix(x,dtype = dtype, type = type, device = device)
  y <- gpu.matrix(y,dtype = dtype, type = type, device = device)
  tol <- 1

  if ((fam == "gaussian") & (link == "identity"))
    maxit <- 1

  while ((tol > acc) & (iter < maxit)) {
    iter <- iter + 1
    beta <- start
    dev0 <- dev
    varmu <- variance(mu)
    mu.eta.val <- mu.eta(as.numeric(eta))
    z <- (eta - offset) + (y - mu)/mu.eta.val
    W <- gpu.matrix((weights * mu.eta.val * mu.eta.val)/varmu,
                    dtype = dtype, type = type, device = device )
    XTX <- crossprod(x*W, x)
    XTz <- t(crossprod((W * z), x))
    # if (iter == 1 & method != "qr") {
    variable <- colnames(x)
    # ris <- if (eigendec)
    #   control(XTX, , tol.values, tol.vectors, , method)
    # else list(rank = nvar, pivot = 1:nvar)
    ris <- list(rank = nvar, pivot = 1:nvar)
    ok <- ris$pivot[1:ris$rank]
    # if (eigendec) {
    #   XTX <- ris$XTX
    #   x <- x[, ok]
    #   XTz <- XTz[ok]
    #   start <- start[ok]
    # }
    beta <- start
    # }
    # if (method == "qr") {
    #   ris <- qr(XTX, tol.values)
    #   ris$coefficients <- drop(qr.solve(ris, XTz, tol.values))
    #   start <- if (ris$rank < nvar)
    #     ris$coefficients[ris$pivot]
    #   else ris$coefficients
    # }
    # else {
    start <- solve(XTX, XTz)
    # }
    eta <- if (sparse)
      drop(x %*% start)
    else drop(tcrossprod(x, t(start)))
    mu <- linkinv(as.numeric(eta <- eta + offset))
    dev <- sum(dev.resids(as.numeric(y), c(mu), weights))
    tol <- max(abs(dev0 - dev)/(abs(dev) + 0.1))
    if (trace)
      cat("iter", iter, "tol", tol, "\n")
  }

  wt <- sum(weights)
  wtdmu <- if (intercept)
    as.numeric(sum(weights * y)/wt)
  else linkinv(offset)
  nulldev <- sum(dev.resids(as.numeric(y), c(wtdmu), weights))
  n.ok <- nobs - sum(weights == 0)
  nulldf <- n.ok - as.integer(intercept)
  rank <- ris$rank
  dfr <- nobs - rank - sum(weights == 0)
  aic.model <- aic(as.numeric(y), nobs, as.numeric(mu), weights, dev) + k * rank
  ll.nuovo <- ll.speedglmGPU(fam, aic.model, rank)
  res <- (y - mu)/mu.eta(as.numeric(eta))
  resdf <- n.ok - rank
  RSS <- sum(W * res * res)
  var_res <- RSS/dfr
  dispersion <- if (fam %in% c("poisson", "binomial"))
    1
  else var_res
  # if (method == "qr") {
  #   coefficients <- start
  #   coefficients[coefficients == 0] = NA
  #   ok <- ris$pivot[1:rank]
  # }
  # else {
  coefficients <- rep(NA, nvar)
  start <- as(start, "numeric")
  coefficients[ok] <- start
  # }
  names(coefficients) <- if (is.null(col.names) & (!is.null(coefficients))) {
    if (intercept) {
      if (length(coefficients) > 1)
        c("(Intercept)", paste("V", 1:(length(coefficients) -
                                         1), sep = ""))
      else "(Intercept)"
    }
    else paste("V", 1:length(coefficients), sep = "")
  }
  else col.names
  rval <- list(coefficients = coefficients, logLik = ll.nuovo,
               iter = iter, tol = tol, family = family, link = link,
               df = dfr, XTX = as.matrix(XTX), dispersion = dispersion, ok = ok,
               rank = rank, RSS = as.numeric(RSS), method = method, aic = aic.model,
               offset = offset, sparse = sparse, deviance = dev, nulldf = nulldf,
               nulldev = nulldev, ngoodobs = n.ok, n = nobs, intercept = intercept,
               convergence = (!(tol > acc)))
  # class(rval) <- "GPUglm"
  return(rval)
}

ll.speedglmGPU <- function (family, aic.model, nvar) {
  switch(family,
         binomial = -(aic.model - 2 * nvar)/2,
         Gamma = -((aic.model - 2 * nvar) - 2)/2,
         gaussian = -((aic.model - 2 * nvar) - 2)/2,
         poisson = -(aic.model - 2 * nvar)/2,
         inverse.gaussian = -((aic.model - 2 * nvar) - 2)/2,
         quasi = NA,
         quasibinomial = NA,
         quasipoisson = NA)
}

# summary.GPUglm <- function (object, correlation = FALSE, ...)
# {
#   if (!inherits(object, "GPUglm"))
#     stop("object is not of class speedglm")
#   z <- object
#   var_res <- as.numeric(z$RSS/z$df)
#   dispersion <- if (z$family$family %in% c("poisson", "binomial")) 1 else var_res
#   if (z$method == "qr") {
#     z$XTX <- z$XTX[z$ok, z$ok]
#   }
#   inv <- solve(z$XTX, tol = z$tol.solve)
#   covmat <- diag(inv)
#   se_coef <- rep(NA, length(z$coefficients))
#   se_coef[z$ok] <- sqrt(dispersion * covmat)
#   if (z$family$family %in% c("binomial", "poisson")) {
#     z1 <- z$coefficients/se_coef
#     p <- 2 * pnorm(abs(z1), lower.tail = FALSE)
#   } else {
#     t1 <- z$coefficients/se_coef
#     p <- 2 * pt(abs(t1), df = z$df, lower.tail = FALSE)
#   }
#   dn <- c("Estimate", "Std. Error")
#   if (z$family$family %in% c("binomial", "poisson")) {
#
#     param <- data.frame(z$coefficients, se_coef, z1,p)
#     dimnames(param) <- list(names(z$coefficients), c(dn,
#                                                      "z value", "Pr(>|z|)"))
#   } else {
#     param <- data.frame(z$coefficients, se_coef, t1,p)
#     dimnames(param) <- list(names(z$coefficients), c(dn,
#                                                      "t value", "Pr(>|t|)"))
#   }
#   eps <- 10 * .Machine$double.eps
#   if (z$family$family == "binomial") {
#     if (any(z$mu > 1 - eps) || any(z$mu < eps))
#       warning("fitted probabilities numerically 0 or 1 occurred")
#   }
#   if (z$family$family == "poisson") {
#     if (any(z$mu < eps))
#       warning("fitted rates numerically 0 occurred")
#   }
#   keep <- match(c("call", "terms", "family", "deviance", "aic",
#                   "df", "nulldev", "nulldf", "iter", "tol", "n", "convergence",
#                   "ngoodobs", "logLik", "RSS", "rank"), names(object),
#                 0)
#   ans <- c(object[keep], list(coefficients = param, dispersion = dispersion,
#                               correlation = correlation, cov.unscaled = inv, cov.scaled = inv *
#                                 var_res))
#   if (correlation) {
#     ans$correl <- (inv * var_res)/outer(na.omit(se_coef),
#                                         na.omit(se_coef))
#   }
#   class(ans) <- "summary.GPUglm"
#   return(ans)
# }


# print.summary.GPUglm <- function (x, digits = max(3, getOption("digits") - 3), ...)
# {
#   cat("Generalized Linear Model of class 'summary.GPUglm':\n")
#   if (!is.null(x$call))
#     cat("\nCall: ", deparse(x$call), "\n\n")
#   if (length(x$coef)) {
#     cat("Coefficients:\n")
#     cat(" ------------------------------------------------------------------",
#         "\n")
#     sig <- function(z){
#       if (!is.na(z)){
#         if (z < 0.001)
#           "***"
#         else if (z < 0.01)
#           "** "
#         else if (z < 0.05)
#           "*  "
#         else if (z < 0.1)
#           ".  "
#         else "   "
#       } else "   "
#     }
#     sig.1 <- sapply(as.numeric(as.character(x$coefficients[,4])),
#                     sig)
#     est.1 <- cbind(format(x$coefficients, digits = digits),
#                    sig.1)
#     colnames(est.1)[ncol(est.1)] <- ""
#     print(est.1)
#     cat("\n")
#     cat("-------------------------------------------------------------------",
#         "\n")
#     cat("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
#         "\n")
#     cat("\n")
#   }
#   else cat("No coefficients\n")
#   cat("---\n")
#   cat("null df: ", x$nulldf, "; null deviance: ", round(x$nulldev,
#                                                         digits = 2), ";\n", "residuals df: ", x$df, "; residuals deviance: ",
#       round(x$deviance, digits = 2), ";\n", "# obs.: ", x$n,
#       "; # non-zero weighted obs.: ", x$ngoodobs, ";\n", "AIC: ",
#       x$aic, "; log Likelihood: ", x$logLik, ";\n", "RSS: ",
#       round(x$RSS, digits = 1), "; dispersion: ", x$dispersion,
#       "; iterations: ", x$iter, ";\n", "rank: ", round(x$rank,
#                                                        digits = 1), "; max tolerance: ", format(x$tol, scientific = TRUE,
#                                                                                                 digits = 3), "; convergence: ", x$convergence, ".\n",
#       sep = "")
#   invisible(x)
#   if (x$correlation) {
#     cat("---\n")
#     cat("Correlation of Coefficients:\n")
#     x$correl[upper.tri(x$correl, diag = TRUE)] <- NA
#     print(x$correl[-1, -nrow(x$correl)], na.print = "", digits = 2)
#   }
# }


# print.GPUglm <- function(x,digits = max(3, getOption("digits") - 3),...)
# {
#   cat("Generalized Linear Model of class 'gpu.matrix':\n")
#   if (!is.null(x$call)) cat("\nCall: ", deparse(x$call), "\n\n")
#   if (length(x$coef)) {
#     cat("Coefficients:\n")
#     print.default(format(x$coefficients, digits = digits), print.gap = 2,
#                   quote = FALSE)
#   } else cat("No coefficients\n")
#   cat("\n")
#   invisible(x)
# }
setMethod("print", signature = "summary.GPUglm", definition = function(x,digits = max(3, getOption("digits") - 3),...){
  cat("Generalized Linear Model of class 'summary.GPUglm':\n")
  if (!is.null(x$call))
    cat("\nCall: ", deparse(x$call), "\n\n")
  if (length(x$coef)) {
    cat("Coefficients:\n")
    cat(" ------------------------------------------------------------------",
        "\n")
    sig <- function(z){
      if (!is.na(z)){
        if (z < 0.001)
          "***"
        else if (z < 0.01)
          "** "
        else if (z < 0.05)
          "*  "
        else if (z < 0.1)
          ".  "
        else "   "
      } else "   "
    }
    sig.1 <- sapply(as.numeric(as.character(x$coefficients[,4])),
                    sig)
    est.1 <- cbind(format(x$coefficients, digits = digits),
                   sig.1)
    colnames(est.1)[ncol(est.1)] <- ""
    print(est.1)
    cat("\n")
    cat("-------------------------------------------------------------------",
        "\n")
    cat("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
        "\n")
    cat("\n")
  }
  else cat("No coefficients\n")
  cat("---\n")
  cat("null df: ", x$nulldf, "; null deviance: ", round(x$nulldev,
                                                        digits = 2), ";\n", "residuals df: ", x$df, "; residuals deviance: ",
      round(x$deviance, digits = 2), ";\n", "# obs.: ", x$n,
      "; # non-zero weighted obs.: ", x$ngoodobs, ";\n", "AIC: ",
      x$aic, "; log Likelihood: ", x$logLik, ";\n", "RSS: ",
      round(x$RSS, digits = 1), "; dispersion: ", x$dispersion,
      "; iterations: ", x$iter, ";\n", "rank: ", round(x$rank,
                                                       digits = 1), "; max tolerance: ", format(x$tol, scientific = TRUE,
                                                                                                digits = 3), "; convergence: ", x$convergence, ".\n",
      sep = "")
  invisible(x)
  if (x$correlation) {
    cat("---\n")
    cat("Correlation of Coefficients:\n")
    x$correl[upper.tri(x$correl, diag = TRUE)] <- NA
    print(x$correl[-1, -nrow(x$correl)], na.print = "", digits = 2)
  }
})

setMethod("print", signature = "GPUglm", definition = function(x,digits = max(3, getOption("digits") - 3),...){
  cat("Generalized Linear Model of class 'gpu.matrix':\n")
  if (!is.null(x$call)) cat("\nCall: ", deparse(x$call), "\n\n")
  if (length(x$coef)) {
    cat("Coefficients:\n")
    print.default(format(x$coefficients, digits = digits), print.gap = 2,
                  quote = FALSE)
  } else cat("No coefficients\n")
  cat("\n")
  invisible(x)
})
setMethod(f ="show", signature = "GPUglm", definition = function(object){
  print(object)
})
setMethod(f ="show", signature = "summary.GPUglm", definition = function(object){
  print(object)
})

setMethod("summary", signature = "GPUglm", definition = function(object, correlation = FALSE, ...){
  if (!inherits(object, "GPUglm"))
    stop("object is not of class speedglm")
  z <- object
  var_res <- as.numeric(z$RSS/z$df)
  dispersion <- if (z$family$family %in% c("poisson", "binomial")) 1 else var_res
  if (z$method == "qr") {
    z$XTX <- z$XTX[z$ok, z$ok]
  }
  inv <- solve(z$XTX, tol = z$tol.solve)
  covmat <- diag(inv)
  se_coef <- rep(NA, length(z$coefficients))
  se_coef[z$ok] <- sqrt(dispersion * covmat)
  if (z$family$family %in% c("binomial", "poisson")) {
    z1 <- z$coefficients/se_coef
    p <- 2 * pnorm(abs(z1), lower.tail = FALSE)
  } else {
    t1 <- z$coefficients/se_coef
    p <- 2 * pt(abs(t1), df = z$df, lower.tail = FALSE)
  }
  dn <- c("Estimate", "Std. Error")
  if (z$family$family %in% c("binomial", "poisson")) {

    param <- data.frame(z$coefficients, se_coef, z1,p)
    dimnames(param) <- list(names(z$coefficients), c(dn,
                                                     "z value", "Pr(>|z|)"))
  } else {
    param <- data.frame(z$coefficients, se_coef, t1,p)
    dimnames(param) <- list(names(z$coefficients), c(dn,
                                                     "t value", "Pr(>|t|)"))
  }
  eps <- 10 * .Machine$double.eps
  if (z$family$family == "binomial") {
    if (any(z$mu > 1 - eps) || any(z$mu < eps))
      warning("fitted probabilities numerically 0 or 1 occurred")
  }
  if (z$family$family == "poisson") {
    if (any(z$mu < eps))
      warning("fitted rates numerically 0 occurred")
  }
  keep <- match(c("call", "terms", "family", "deviance", "aic",
                  "df", "nulldev", "nulldf", "iter", "tol", "n", "convergence",
                  "ngoodobs", "logLik", "RSS", "rank"), names(object),
                0)
  ans <- c(object[keep], list(coefficients = param, dispersion = dispersion,
                              correlation = correlation, cov.unscaled = inv, cov.scaled = inv *
                                var_res))
  if (correlation) {
    ans$correl <- (inv * var_res)/outer(na.omit(se_coef),
                                        na.omit(se_coef))
  }
  class(ans) <- "summary.GPUglm"
  return(ans)
})

