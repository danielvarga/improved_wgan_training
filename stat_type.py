# return a the type of the record
# should return None if the record is to be filtered out
def get_type(record):
    return compare_mnist_1(record)

def compare_mnist_1_tune(record):
    if record['bn'] == "y":
        return None
    elif record['do'] == 1:
        type = 'no'
    else:
        type = 'do'
    if record['lambda'] > 0:
        return None
    return type

def compare_mnist_1(record):
    if record['bn'] == "y":
        if record['dg'] == 0.001:
            return "DataGrad"
        elif record['lambda'] == 0.001:
            return "SpecReg"
        elif record['lambda'] == 0 and record['dg'] == 0:
            return "Unreg"
        else:
            return None

    if record['dg'] == 50:
        return "DataGrad"
    elif record['lambda'] == 0.01:
        return "SpecReg"
    elif record['lambda'] == 0 and record['dg'] == 0:
        return "Unreg"
    return None

def compare_mnist_1b(record):
    if record['bn'] == "y":
        if record['dg'] == 0.003:
            return "DataGrad"
        elif record['lambda'] == 0.003:
            return "SpecReg"
        elif record['lambda'] == 0 and record['dg'] == 0:
            return "Unreg"
        else:
            return None

    if record['dg'] == 50:
        return "DataGrad"
    elif record['lambda'] == 0.01:
        return "SpecReg"
    elif record['lambda'] == 0 and record['dg'] == 0:
        return "Unreg"
    return None


def compare_mnist_2(record):
    # compare unreg, datagrad, ent, ent+datagrad
    if record['dg'] != 0 and record['ent'] > 0:
        type = "ent+dg"
    elif record['dg'] != 0:
        type = "dg"
    elif record['ent'] > 0:
        type = "ent"
    else:
        type = "unreg"
    return type
    
    
"""
                
        if False: # comparing datagrad parameters with batchnorm
            if record['dg'] == 0 or record['bn'] != "y":
                continue
            type = "dg_bn_" + str(record['dg'])
        elif False: # comparing datagrad parameters with dropout
            if record['dg'] == 0 or record['bn'] == "y":
                continue
            type = "dg_do_" + "%05.2f" % record['dg']
        elif False: # comparing datagrad bn with datagrad do
            if record['dg'] == 0:
                continue
            type = "dg_bn_" + str(record['bn'])
        elif False: # comparing unreg dropout and batchnorm
            if record['dg'] != 0 or record['lambda'] != 0 or record['gs'] == 'y':
                continue
            type = "unreg_bn_" + record['bn']
        elif False: # comparing GP with L2 gradient loss bn vs dropout
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 3:
                continue
            type = "gp3a_bn_" + record['bn']
        elif False: # comparing GP with L2 gradient loss with dropout for various lambdas
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 3 or record['bn'] != "n":
                continue
            type = "gp3a_do_" + "%06.4f" % record['lambda']
        elif False: # comparing GP with softmax bn vs dropout
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "softmax" or record['gp'] != 3:
                continue
            type = "gp4_bn_" + record['bn']
        elif False: # comparing GP with softmax for various lambdas
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "softmax" or record['gp'] != 3:
                continue
            type = "gp4_" + "%06.4f" % record['lambda']
        elif False: # comparing GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET bn vs dropout
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 4:
                continue
            type = "gp3b_bn_" + record['bn']
        elif False: # comparing GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET with dropout for various lipschitz_targets
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 4  or record['bn'] != "n":
                continue
            if record['lips'] > 2:
                continue
            type = "gp3b_do_" + "%05.2f" % record['lips']
        elif False: # comparing GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET with dropout for LIPS=0.7 various lambdas
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 4  or record['bn'] != "n" or record['lips'] != 0.7:
                continue
            type = "gp3b_do_lips_0.7_" + "%06.4f" % record['lambda']
        elif False: # final comparison
            if record['dg'] == 0 and record['lambda'] == 0 and record['gs'] == 'n' and record['bn'] == 'n':
                type = "1_unreg"
            elif record['dg'] == 10 and record['bn'] == "n":
                type = "2_datagrad"
            elif record['dg'] == 0 and record['lambda'] == 0.01 and record['comb'] == "random" and record['gp'] == 3 and record['bn'] == 'n':
                type = "3a_gp_to_zero"
            elif record['dg'] == 0 and record['lambda'] == 0.01 and record['comb'] == "random" or record['gp'] == 4  and record['bn'] == "n" and record['lips'] == 0.7:
                type = "3b_gp_to_lips"
            elif record['dg'] == 0 and record['lambda'] == 0.1 and record['comb'] == "softmax" and record['gp'] == 3:
                type = "4_softmax"
            else:
                continue
        elif True: # final comparison
            type = "lenet"
            type += "_comb" + str(record['comb'])
            type += "_lambda" + str(record['lambda'])
            type += "_ent" + str(record['ent'])
            type += "_bn" + str(record['bn'])
            type += "_wd" + str(record['wd'])
            type += "_net" + str(record['net'])

        else:
            type = "unknown"
            if record['lambda'] == 0 and 'dg' in record  and  record['dg'] == 0:
                type = 'nogp'
            elif record['lambda'] != 0 and record['gp'] == 2:
                type = 'gp'
            elif record['lambda'] == 0 and 'dg' in record and record['dg'] != 0:
                type = 'datagrad'
            elif record['lambda'] == 0 and record['gs'] == 'y':
                type = 'gs'

"""
