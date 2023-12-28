import os
import numpy as np
import torch
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as robjects

importr('openEBGM')


def check_and_make_path(to_make):
    if to_make == '':
        return
    if not os.path.exists(to_make):
        os.makedirs(to_make)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data).float()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def ror_ebgm(drug_reac_num, theta_init):

    def signal_discrete(x, x_025, y, y_025):
        if (x_025 <= 1) | (y_025 <= 1):
            if (x > 1) & (y > 1):
                return 2
            else:
                return 1
        else:
            if (x > 30) & (y > 30):
                return 5
            elif (x > 5) & (y > 5):
                return 4
            else:
                return 3

    ror = drug_reac_num.copy()

    ror['b'] = ror['drug_num'] - ror['a']
    ror['c'] = ror['reac_num'] - ror['a']
    ror['d'] = ror['n'] - ror['b'] - ror['c'] - ror['a']
    ror['T_a'] = ror['drug_num'] * ror['reac_num'] / ror['n']

    ror = ror[(ror['b'] > 0) & (ror['c'] > 0)]

    ror['ror'] = ror['a'] * ror['d'] / (ror['c'] * ror['b'])
    temp = np.sqrt(1 / ror['a'] + 1 / ror['b'] + 1 / ror['c'] + 1 / ror['d'])
    ror['ror_025'] = np.exp(np.log(ror['ror']) - 1.96 * temp)

    ror = ror[['Drug', 'Reac', 'a', 'b', 'c', 'd', 'T_a', 'ror', 'ror_025']] \
        .rename(columns={'Drug': 'var1', 'Reac': 'var2', 'a': 'N', 'T_a': 'E'})

    # EBGM calculation
    if ror.empty == False:

        with localconverter(robjects.default_converter + pandas2ri.converter):
            ror_NE = robjects.conversion.py2rpy(ror)
            theta_init_r = robjects.conversion.py2rpy(theta_init)

        robjects.globalenv['ror_NE'] = ror_NE
        robjects.globalenv['theta_init_r'] = theta_init_r

        theta_hat = robjects.r("""
            flag <- 0
            squashed <- autoSquash(ror_NE)
            for (method in c('nlminb', 'nlm', 'bfgs')){
                suppressWarnings(theta_hat <- exploreHypers(data=squashed, theta_init=theta_init_r, max_pts=50000, N_star=1, method=method)$estimates)

                conv <- theta_hat$converge == TRUE & !is.na(theta_hat$converge)
                within_bounds <- theta_hat$in_bounds == TRUE & !is.na(theta_hat$in_bounds)
                theta_conv <- theta_hat[conv & within_bounds, ]
                if (dim(theta_conv)[1] >= 1){
                    theta_hat_final <- theta_conv[theta_conv$minimum == min(theta_conv$minimum), 2:6]
                    flag = 1
                    break
                }
            }
            c(flag, theta_hat_final)
        """)

        flag = np.array(np.array(theta_hat).reshape((1, 6))).T[0]

        if flag == 1:
            theta_hat = np.array(theta_hat).T[0][1:]
            if theta_init.shape[0] == 4:
                t = pd.DataFrame(theta_hat).transpose()
                t.columns = theta_init.columns
                theta_init = pd.concat([t, theta_init])
            else:
                theta_init.iloc[0, :] = theta_hat

            robjects.globalenv['theta_hat_final'] = robjects.FloatVector(theta_hat)

            ebgm = robjects.r("""
                  qn <- Qn(theta_hat_final, N=ror_NE$N, E= ror_NE$E)
                  EBGM <- ebgm(theta_hat_final, N=ror_NE$N, E= ror_NE$E, qn=qn)
                  QUANT_025 <- quantBisect(percent=2.5, theta=theta_hat_final, N=ror_NE$N, E=ror_NE$E, qn=qn)
                  out <- data.frame(EBGM, QUANT_025)
            """)

            ror['EBGM'] = np.array(ebgm.rx('EBGM')).T
            ror['EBGM_025'] = np.array(ebgm.rx('QUANT_025')).T
            ror['ROR_EBGM'] = ror['ror'] * ror['EBGM']
            ror = ror.rename(columns={'var1': 'Drug', 'var2': 'Reac', 'ror': 'ROR', 'ror_025': 'ROR_025'})

            ror['signal'] = ror.apply(lambda x: signal_discrete(x['ROR'], x['ROR_025'], x['EBGM'], x['EBGM_025']), axis=1)

        else:
            print("records: {}".format("Unable to converge"))
    else:
        print("records: {}".format('ror is empty'))

    return ror[['Drug', 'Reac', 'signal', 'ROR', 'EBGM', 'N', 'ROR_025', 'EBGM_025', 'b', 'c', 'd']], theta_init
