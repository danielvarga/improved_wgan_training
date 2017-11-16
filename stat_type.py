# return a the type of the record
# should return None if the record is to be filtered out
def get_type(record, type_grouping):
    if type_grouping == None:
        return basic_types(record)
    elif type_grouping == "datagrad_lenet":
        return types_datagrad_lenet(record)
    elif type_grouping == "datagrad_lenettuned":
        return types_datagrad_lenettuned(record)
    elif type_grouping == "entropy_lenet":
        return types_entropy_lenet(record)
    elif type_grouping == "entropy_lenettuned":
        return types_entropy_lenettuned(record)
    elif type_grouping == "gp_lenet":
        return types_gp_lenet(record)
    elif type_grouping == "gp_lenettuned":
        return types_gp_lenettuned(record)
    elif type_grouping == "onehot_lenet":
        return types_onehot_lenet(record)
    elif type_grouping == "onehot_lenettuned":
        return types_onehot_lenettuned(record)
    elif type_grouping == "jacreg_lenet":
        return types_jacreg_lenet(record)
    elif type_grouping == "jacreg_lenettuned":
        return types_jacreg_lenettuned(record)

    else:
        assert False

# this is called when no type grouping is specified
def basic_types(record):
    type = "net_" + str(record['net'])
    type += "-comb_" + str(record['comb'])
    type += "-lambda_" + str(record['lambda'])
    type += "-ent_" + str(record['ent'])
    type += "-bn_" + str(record['bn'])
    type += "-wd_" + str(record['wd'])
    return type

# for visualizing grid_mnist_datagrad.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_datagrad accuracy test 10000 -type_grouping datagrad_lenet
def types_datagrad_lenet(record):
    if record['net'] != "lenet":
        return None
    elif record['dg'] >= 100: # these are all worse results
        return None
    else:
        return "dg_%03.0f" % record['dg']

# for visualizing grid_mnist_datagrad.sh results for lenettuned
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_datagrad accuracy test 10000 -type_grouping datagrad_lenettuned
def types_datagrad_lenettuned(record):
    if record['net'] != "lenettuned":
        return None
    elif record['dg'] >= 100: # these are all worse results
        return None
    else:
        return "dg_%03.0f" % record['dg']

# for visualizing grid_mnist_ent.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_ent accuracy test 10000 -type_grouping entropy_lenet
def types_entropy_lenet(record):
    if record['net'] != "lenet":
        return None
    else:
        return "ent %05.3f" % record['ent']

# for visualizing grid_mnist_ent.sh results for lenettuned
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_ent accuracy test 10000 -type_grouping entropy_lenettuned
def types_entropy_lenettuned(record):
    if record['net'] != "lenettuned":
        return None
    elif record['ent'] <= 0.003: # these are all worse results
        return None
    else:
        return "ent %04.2f" % record['ent']

# for visualizing grid_mnist_gp.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_gp accuracy test 10000 -type_grouping gp_lenet
def types_gp_lenet(record):
    if record['net'] != "lenet":
        return None
    elif record['lambda'] >= 0.1 or record['lambda'] <= 0.0001:
        return None
    else:
        return "lambda %06.4f" % record['lambda']

# for visualizing grid_mnist_gp.sh results for lenettuned
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_gp accuracy test 10000 -type_grouping gp_lenettuned
def types_gp_lenettuned(record):
    if record['net'] != "lenettuned":
        return None
    elif record['lambda'] >= 0.1 or record['lambda'] <= 0.0001:
        return None
    else:
        return "lambda %06.4f" % record['lambda']

# for visualizing grid_mnist_jacreg.sh results for lenet
# python stat.py /home/zombori/working_shadow/logs accuracy test 10000 -type_grouping jacreg_lenet
def types_jacreg_lenet(record):
    if record['net'] != "lenet":
        return None
#    elif record['lambda'] >= 0.1 or record['lambda'] <= 0.0001:
#        return None
    else:
        return "lambda %06.4f" % record['lambda']

# for visualizing grid_mnist_jacreg.sh results for lenettuned
# python stat.py /home/zombori/working_shadow2/logs accuracy test 10000 -type_grouping jacreg_lenettuned
def types_jacreg_lenettuned(record):
    if record['net'] != "lenettuned":
        return None
#    elif record['lambda'] >= 0.1 or record['lambda'] <= 0.0001:
#        return None
    else:
        return "lambda %06.4f" % record['lambda']


# for visualizing grid_mnist_onehot.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_onehot accuracy test 10000 -type_grouping onehot_lenet
def types_onehot_lenet(record):
    if record['net'] != "lenet":
        return None
    # elif record['lambda'] >= 0.1 or record['lambda'] <= 0.0001:
    #     return None
    else:
        return "lambda %06.4f" % record['lambda']

    
# for visualizing grid_mnist_onehot.sh results for lenettuned
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_onehot accuracy test 10000 -type_grouping onehot_lenettuned
def types_onehot_lenettuned(record):
    if record['net'] != "lenettuned":
        return None
    # elif record['lambda'] >= 0.1 or record['lambda'] <= 0.0001:
    #     return None
    else:
        return "lambda %06.4f" % record['lambda']


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
