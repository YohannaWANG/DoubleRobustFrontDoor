library(colordistance) # Chi-square distance 
# colordistance:::chisqDistance(rnorm(10), rnorm(10))

library(readr)
library(foreach)
library(MASS)
library(doParallel)
library(magrittr)
library(numDeriv)

cl <- makeCluster(10) 
registerDoParallel(cl)


load("truth/frontdoor-truth.rda")



S = 10
n= 10000

######## GENERATE DATASET ###########

expit <- function(x){
  output <- exp(x) / (1 + exp(x))
  return(output)
}
# n,p.c1,p.c3,alpha.truth,omega.truth,beta.truth,theta.truth,sigma.m,sigma.y

gen.med.data.continuous <- function(n,p.c1,p.c3,alpha,omega,beta,theta,sigma.m,sigma.y){
  
  #Generate first confounder 
  c1 <- rbinom(n,1,p.c1)
  
  #Generate second confounder (given c1)
  p.c2 <- expit(cbind(rep(1,n),c1)%*%alpha)*(1-as.numeric(alpha[1] == 0 &  alpha[2] == 0)) 
  c2 <- rbinom(n,1,p.c2) 
  
  #Generate a confounder of A & Y (independent of other confounders)
  c3 <- rbinom(n,1,p.c3)
  
  #Generate binary exposure (given c1, c2 and c1*c2)
  p.a <- expit(cbind(rep(1,n),c1,c2,c1*c2,c3)%*%omega)
  a <- rbinom(n,1,p.a) 
  
  #Generate continuous mediator (given a, c1, c2 and all possible combinations)
  mean.m <- cbind(rep(1,n),a,c1,c2,c1*c2,c3)%*%beta
  m <- rnorm(n,mean.m,sigma.m)
  
  #Generate continuous outcome Y (given a,m,c1,c2, c3 and possible combos)
  mean.y <- cbind(rep(1,n),a,m,a*m,c1,c2,c1*c2,c3)%*%theta
  y <- rnorm(n,mean.y,sigma.y)
  
  sim.data <- data.frame(cbind(y,a,m,c1,c2,c3))
  
  return(sim.data)
  
}

###### FUNCTION THAT CALCULATE PSI DOUBLY ROBUST SP###### 

piie.sp.variance.function.cont <- function(cov.vals.all,exposure,intermediate,outcome,i.y,i.z,i.a,fit.a,fit.z,fit.y,astar,interaction){
  
  n <- length(exposure)
  
  sigma <- summary(fit.z)$sigma
  
  model.matrix.a <- as.matrix(data.frame(model.matrix(fit.a)))
  model.matrix.z <- as.matrix(data.frame(model.matrix(fit.z)))
  model.matrix.y <- as.matrix(data.frame(model.matrix(fit.y)))
  
  model.matrix.z_astar <- data.frame(model.matrix.z)
  model.matrix.z_astar[,2] <- 0
  model.matrix.z_astar <- as.matrix(model.matrix.z_astar)
  
  theta.hat <- summary(fit.y)$coefficients[,1]
  beta.hat <- summary(fit.z)$coefficients[,1]
  alpha.hat <- summary(fit.a)$coefficients[,1]
  
  cov.vals.y <- cov.vals.all[,which(i.y==1)]
  cov.vals.z <- cov.vals.all[,which(i.z==1)]
  cov.vals.a <- cov.vals.all[,which(i.a==1)]
  
  z.mean_astar <- model.matrix.z_astar%*%beta.hat
  z.mean_ind <- model.matrix.z%*%beta.hat
  a.mean <- expit(model.matrix.a%*%alpha.hat)
  y.mean <- model.matrix.y%*%theta.truth#theta.hat
  
  if (interaction == 1){sum.a <- as.matrix(cbind(rep(1,n),a.mean,model.matrix.y[,3],a.mean*model.matrix.y[,3],cov.vals.y))%*%theta.truth#theta.hat
  } else { sum.a <- as.matrix(cbind(rep(1,n),a.mean,model.matrix.y[,3],cov.vals.y))%*%theta.hat }
  
  if (interaction == 1){sum.z <- as.matrix(cbind(rep(1,n),model.matrix.y[,2],z.mean_astar,model.matrix.y[,2]*z.mean_astar,cov.vals.y))%*%theta.truth#theta.hat
  } else { sum.z <- as.matrix(cbind(rep(1,n),model.matrix.y[,2],z.mean_astar,cov.vals.y))%*%theta.hat}
  
  if (interaction == 1){sum.az <- as.matrix(cbind(rep(1,n),a.mean,z.mean_ind,a.mean*z.mean_ind,cov.vals.y))%*%theta.truth#theta.hat
  } else {sum.az <- as.matrix(cbind(rep(1,n),a.mean,z.mean_ind,cov.vals.y))%*%theta.hat}
  
  psi.sp.ind <-  ((outcome - y.mean)*
                    (dnorm(model.matrix.y[,3],z.mean_astar,sigma)/dnorm(model.matrix.y[,3],z.mean_ind,sigma))
                  + ((1-model.matrix.y[,2])/(1-a.mean))*(sum.a - sum.az) 
                  + sum.z )
  
  piie.sp.ind <- outcome - psi.sp.ind
  
  piie.sp <- mean(outcome) - mean(psi.sp.ind)
  
  score.sp <- cbind( model.matrix.a*c(exposure - a.mean),
                     model.matrix.z*c(intermediate - z.mean_ind),
                     model.matrix.y*c(outcome - y.mean),
                     (piie.sp.ind - piie.sp))
  
  estimates <- c(alpha.hat,beta.hat,theta.hat)
  len.a <- length(coefficients(fit.a))
  len.z <- length(coefficients(fit.z))
  
  #take the derivative and plugs in inputs
  deriv.sp <- numDeriv::jacobian(U.sp,c(estimates,piie.sp),model.matrix.a=model.matrix.a,model.matrix.z=model.matrix.z,model.matrix.y=model.matrix.y,data.y=outcome,data.z=intermediate,data.a=exposure,i.y=i.y,i.z=i.z,i.a=i.a,cov.vals.all=cov.vals.all,len.a=len.a,len.z=len.z,n=n,sigma=sigma,interaction=interaction)
  
  #Calculate variance matrix
  var.sp <- (solve(deriv.sp)%*%t(score.sp)%*%score.sp%*%t(solve(deriv.sp)))
  
  #Variance for our estimator 
  piie.var.sp <- var.sp[length(c(estimates,piie.sp)),length(c(estimates,piie.sp))]
  
  #Chi-square distance measure
  # sigma_y = 1 
  distance <- mean(exp((model.matrix.y%*%theta.truth -  model.matrix.y%*%theta.hat)^2) - 1)
  
  output <- cbind(mean(psi.sp.ind), piie.sp, piie.var.sp, distance)
  colnames(output) <- c("PSI","PIIE","Var PIIE","Chisquare")
  
  return(output)
  
}

U.sp <- function(estimates,model.matrix.a,model.matrix.z,model.matrix.y,data.a,data.z,data.y,i.y,i.z,i.a,cov.vals.all,len.a,len.z,n,sigma,interaction){
  
  alpha.hat <- estimates[1:len.a]
  beta.hat <- estimates[(len.a+1):(len.a+len.z)]
  theta.hat <- estimates[(len.a+len.z+1):(length(estimates) - 1)]
  piie.est <- estimates[length(estimates)]
  
  model.matrix.z_astar <- data.frame(model.matrix.z)
  model.matrix.z_astar[,2] <- 0
  model.matrix.z_astar <- as.matrix(model.matrix.z_astar)
  
  cov.vals.y <- cov.vals.all[,which(i.y==1)]
  cov.vals.z <- cov.vals.all[,which(i.z==1)]
  cov.vals.a <- cov.vals.all[,which(i.a==1)]
  
  z.mean_astar <- model.matrix.z_astar%*%beta.hat
  z.mean_ind <- model.matrix.z%*%beta.hat
  a.mean <- expit(model.matrix.a%*%alpha.hat)
  y.mean <- model.matrix.y%*%theta.hat
  
  if (interaction == 1){sum.a <- as.matrix(cbind(rep(1,n),a.mean,model.matrix.y[,3],a.mean*model.matrix.y[,3],cov.vals.y))%*%theta.hat
  } else { sum.a <- as.matrix(cbind(rep(1,n),a.mean,model.matrix.y[,3],cov.vals.y))%*%theta.hat }
  
  if (interaction == 1){sum.z <- as.matrix(cbind(rep(1,n),model.matrix.y[,2],z.mean_astar,model.matrix.y[,2]*z.mean_astar,cov.vals.y))%*%theta.hat
  } else { sum.z <- as.matrix(cbind(rep(1,n),model.matrix.y[,2],z.mean_astar,cov.vals.y))%*%theta.hat }
  
  if (interaction == 1){sum.az <- as.matrix(cbind(rep(1,n),a.mean,z.mean_ind,a.mean*z.mean_ind,cov.vals.y))%*%theta.hat
  } else {sum.az <- as.matrix(cbind(rep(1,n),a.mean,z.mean_ind,cov.vals.y))%*%theta.hat}
  
  psi.sp.ind <- ( (data.y - y.mean)*
                    (dnorm(model.matrix.y[,3],z.mean_astar,sigma)/dnorm(model.matrix.y[,3],z.mean_ind,sigma))
                  + ((1-model.matrix.y[,2])/(1-a.mean))*(sum.a - sum.az) 
                  + sum.z )
  
  piie.sp.ind <- data.y - psi.sp.ind
  
  piie.sp <- mean(data.y) - mean(psi.sp.ind)
  
  score.sp <- cbind( model.matrix.a*c(data.a - a.mean),
                     model.matrix.z*c(data.z - z.mean_ind),
                     model.matrix.y*c(data.y - y.mean),
                     (piie.sp.ind - piie.est))
  
  deriv <- matrix(1,1,n)%*%score.sp
  
  return(deriv)
}

#####################################
# SIMULATIONS -- CLUSTER -- CORRECT #
#####################################

parOut <- foreach(s=1:S) %dopar% {
  
  data <- gen.med.data.continuous(n,p.c1,p.c3,alpha.truth,omega.truth,beta.truth,theta.truth,sigma.m,sigma.y)
  
  fit.z <- lm(m ~ a + c1 + c2 + I(c1*c2),data=data) 
  fit.y <- lm(y ~ a + m + I(a*m) + c1 + c2 + I(c1*c2) + c3,data=data) 
  fit.a <- glm(a ~ c1 + c2 + I(c1*c2) + c3,data=data,family=binomial)
  
  alpha.hat <- summary(fit.a)$coefficients[,1]
  beta.hat <- c(summary(fit.z)$coefficients[,1],0)
  theta.hat <- summary(fit.y)$coefficients[,1]

  confounders <- cbind(data[,4:5],data[,4]*data[,5],data[,6])

  
  # DR  --- sim-correct-scatter #
  out.sp <- piie.sp.variance.function.cont(confounders,data$a,data$m,data$y,c(1,1,1,1),c(1,1,1,0),c(1,1,1,1),fit.a,fit.z,fit.y,0,1)
  prop.bias.sp <- (out.sp[2]-truth.est[3])/truth.est[3]
  
  distance <- out.sp[4]
  
  ci.low <- out.sp[2] - qnorm(.975)*sqrt(out.sp[3])
  ci.up <- out.sp[2] + qnorm(.975)*sqrt(out.sp[3])
  ci.coverage.sp <- as.numeric(truth.est[3] >= ci.low & truth.est[3] <= ci.up)
  
  out.sp <- cbind(out.sp,prop.bias.sp,ci.coverage.sp)
  
  mean.y <- mean(data$y)
  
  list(out.sp, mean.y, distance)
}

getter <- function(iter, index) iter[[index]]


sp <- sapply(parOut, getter, 1) %>% t()
mean.y <- sapply(parOut, getter, 2) %>% t()
distance <- sapply(parOut, getter, 3) %>% t()# 

#distance<- as.numeric(distance)
#distance.low <- mean(distance) - 2*(mean(distance))
#distance.up  <- mean(distance) + 2*(mean(distance))
#distance_filter <- subset(distance, distance >= distance.low & distance <= distance.up)

outlist <- list(sp, mean.y, distance)
# saveRDS(outlist, file = paste0("output/out_correct.rds"))

