# return a the type of the record
# should return None if the record is to be filtered out
def get_type(record, type_grouping):
    if type_grouping == None:
        return basic_types(record)

    type_function = "types_" + type_grouping
    try:
        fun = globals()[type_function]
    except:
        print "Unrecognized type_grouping " + type_grouping + ", using function basic_types instead of " + type_function
        return basic_types(record)
    return fun(record)


# this is called when no type grouping is specified
def basic_types(record):
    type = "net_" + str(record['net'])
    type += "-comb_" + str(record['comb'])
    type += "-lambda_" + str(record['lambda'])
    type += "-ent_" + str(record['ent'])
    type += "-bn_" + str(record['bn'])
    type += "-wd_" + str(record['wd'])
    return type

# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_full accuracy test 50000 -type_grouping mnist_full
def types_mnist_full(record):
    if record['dg'] > 0:
        if record['ent'] > 0:
            return "ent+dg"
        else:
            return "dg"
    elif record['ent'] > 0:
        return "ent"
    else:
        return "unreg"

# for visualizing grid_cifar10_ent_dg_spect.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar10_ent_dg_spect test_accuracy test 50000 -type_grouping cifar10_ent_dg_spect
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar10_ent_dg_spect_rerun test_accuracy test 50000 -type_grouping cifar10_ent_dg_spect
def types_cifar10_ent_dg_spect(record):
    if record['ent'] > 0:
        return "EntReg"
    elif record['dg'] > 0:
        return "DataGrad"
    elif record['lambda'] > 0:
        return "SpectReg"
    else:
        return "Baseline"

# for visualizing grid_cifar10_dg_spect_wd.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar10_dg_spect_wd test_accuracy test 50000 -type_grouping cifar10_dg_spect_wd -x_key wd
def types_cifar10_dg_spect_wd(record):
    if record['dg'] > 0:
        prefix = "DataGrad"
    elif record['lambda'] > 0:
        prefix = "SpectReg"
    else:
        prefix = "Baseline"
    return prefix + "_wd-{}".format(record['wd'])

# for visualizing grid_cifar10_spectreg_nowd.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar10_spectreg_nowd devel_accuracy test 50000 -type_grouping cifar100_spectreg -x_key lambda

# for visualizing grid_mnist_4.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_4 test_accuracy test 10000 -type_grouping mnist_4
def types_mnist_4(record):
    if record['combslopes'] == 'n':
        return "JacReg"
    elif record['lambda'] > 0:
        return "SpectReg"
    elif record['dg'] > 0:
        return "DoubleBack"
    elif record['ent'] > 0:
        return "Confidence Penalty"
    else:
        return "Baseline (weight decay)"

# for visualizing grid_mnist_6.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_6 test_accuracy test 10000 -type_grouping mnist_6
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_6_rerun test_accuracy test 10000 -type_grouping mnist_6
def types_mnist_6(record):
    if record['comb'] == 'softmax':
        return "JacReg"
    elif record['comb'] == 'onehot':
        return "Onehot"
    elif record['comb'] == 'random_onehot':
        return "Random Onehot"    
    elif record['combslopes'] == 'n':
        if record['lambda'] == 0.03:
            return "Frob"
        else:
            return None
    elif record['lambda'] > 0:
        return "SpectReg"
    elif record['dg'] > 0:
        return "DataGrad"
    else:
        return "blabla"

# for visualizing grid_cifar10_6.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar10_6 test_accuracy test 50000 -type_grouping cifar10_6
def types_cifar10_6(record):
    if record['comb'] == 'softmax':
        return "JacReg"
    elif record['comb'] == 'onehot':
        return "Onehot"
    elif record['comb'] == 'random_onehot':
        return "Random Onehot"    
    elif record['combslopes'] == 'n':
        if record['lambda'] == 0.03:
            return "Frob"
        else:
            return None
    elif record['lambda'] > 0:
        return "SpectReg"
    elif record['dg'] > 0:
        return "DataGrad"
    else:
        return "blabla"

# for visualizing grid_cifar10_frobreg.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar10_frobreg devel_accuracy test 50000 -type_grouping cifar10_frobreg
def types_cifar10_frobreg(record):
    return "FrobReg_{0:0f}".format(record['lambda'])


# for visualizing grid_cifar100_spectreg.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar100_spectreg test_accuracy test 50000 -type_grouping cifar100_spectreg -x_key lambda
def types_cifar100_spectreg(record):
    return "SpectReg_{0:0f}".format(record['lambda'])

# for visualizing grid_cifar100_datagrad.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar100_datagrad test_accuracy test 50000 -type_grouping cifar100_dg -x_key dg
def types_cifar100_dg(record):
    return "DataGrad_{0:0f}".format(record['dg'])

# for visualizing grid_cifar100_dg_spect.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar100_dg_spect test_accuracy test 50000 -type_grouping cifar100_dg_spect_ent
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar100_dg_spect_aug test_accuracy test 50000 -type_grouping cifar100_dg_spect_ent

# for visualizing grid_cifar100_dg_spect.sh results
# for visualizing grid_cifar100_entreg_repeated.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar100_dg_spect_ent test_accuracy test 50000 -type_grouping cifar100_dg_spect_ent
def types_cifar100_dg_spect_ent(record):
    if record['dg'] > 0:
        return "DataGrad"
    elif record['lambda'] > 0:
        return "SpectReg"
    elif record['ent'] > 0:
        return "EntReg"
    else:
        return "Baseline"

# for visualizing grid_cifar100_entreg.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/cifar100_entreg devel_accuracy test 50000 -type_grouping cifar100_entreg -x_key ent
def types_cifar100_entreg(record):
    return "EntReg_{0:0f}".format(record['ent'])











# for visualizing grid_mnist_compare2000_onehot.sh results
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_compare2000_onehot test_accuracy test 10000 -type_grouping random_onehot_lenet -x_key net
def types_random_onehot_lenet(record):
    net = record['net']
    comb = record['comb']
    lb = record['lambda']
    if net == "lenet" and lb != 0.3:
        return None
    if net == "lenettuned" and lb != 0.1:
        return None
    return comb + "-" + str(lb)

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

# for visualizing grid_mnist_1_new.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_1 test_accuracy test 10000 -type_grouping mnist_1 -x_key reg
def types_mnist_1(record):
    return record['reg'] + "_dg-" + str(record['dg']) + "_spect-" + str(record['lambda'])


# for visualizing grid_mnist_1b.sh results for lenettuned
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_1b accuracy test 10000 -type_grouping mnist_1b -x_key reg
def types_mnist_1b(record):
    if record['bn'] == "y":
        if record['dg'] == 0.003:
            return "DataGrad"
        elif record['lambda'] == 0.003:
            return "SpectReg"
        elif record['lambda'] == 0 and record['dg'] == 0:
            return "NoGR"
        else:
            return None

    if record['dg'] == 50:
        return "DataGrad"
    elif record['lambda'] == 0.01:
        return "SpectReg"
    elif record['lambda'] == 0 and record['dg'] == 0:
        return "NoGR"
    return None



# for visualizing grid_mnist_2.sh results for lenet
# python stat.py /mnt/g2big/tensorboard_logs/paper1/mnist_2 test_accuracy test 10000 -type_grouping mnist_2
def types_mnist_2(record):
    if record['dg'] > 0 and record['ent'] > 0:
        return "ent+dg"
    elif record['lambda'] > 0 and record['ent'] > 0:
        return "ent+spect"
    elif record['dg'] > 0:
        return "dg"
    elif record['ent'] > 0:
        return "ent"
    elif record['lambda'] > 0:
        return "spect"
    else:
        return "unreg"
    
