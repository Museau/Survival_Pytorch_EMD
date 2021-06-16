#!/usr/bin/env Rscript


# Ref: Venables, W. N. and Ripley, B. D. (2002) Modern Applied Statistics with
#      S. Fourth edition. Springer.
# Adaptation: Margaux Luck <margaux.luck@gmail.com>
# Language: R


Aids2 <- read.csv('Aids2.csv')

time.depend.covar <- function(data) {
  id <- row.names(data)
  n <- length(id)
  events <- c(0, 10043, 11139, 12053)  # julian days
  crit1 <- matrix(events[1:3], n, 3, byrow = T)
  crit2 <- matrix(events[2:4], n, 3, byrow = T)
  diag <- matrix(data$diag, n, 3)
  death <- matrix(data$death, n, 3)
  incid <- (diag < crit2) & (death >= crit1)
  incid <- t(incid)
  indr <- col(incid)[incid]
  indc <- row(incid)[incid]
  ind <- cbind(indr, indc)
  idno <- id[indr]
  state <- data$state[indr]
  T.categ <- data$T.categ[indr]
  age <- data$age[indr]
  sex <- data$sex[indr]
  late <- indc - 1
  start <- t(pmax(crit1 - diag, 0))[incid]
  stop <- t(pmin(crit2, death + 0.9) - diag)[incid]
  status <- matrix(as.numeric(data$status), n, 3) - 1  # 0/1
  status[death > crit2] <- 0
  status <- status[ind]
  levels(state) <- c("NSW", "Other", "QLD", "VIC")
  levels(T.categ) <- c("hs", "hsid", "id", "het", "haem",
                       "blood", "mother", "other")
  levels(sex) <- c("F", "M")
  data.frame(idno, zid=factor(late), start, stop, status,
             state, T.categ, age, sex)
}

Aids3 <- time.depend.covar(Aids2)

write.csv(Aids3, file='Aids3.csv', row.names=FALSE)
