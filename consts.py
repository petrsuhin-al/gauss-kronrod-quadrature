from numpy import array, negative, append, concatenate

# узлы и веса для Гаусса-Кронрода

gausskronrod_nodes = {
    15: array([
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245, 0,
    ]),
    21: array([
        0.99565716302580809, 0.97390652851717174,
        0.93015749135570824, 0.86506336668898454,
        0.7808177265864169, 0.67940956829902444,
        0.56275713466860466, 0.43339539412924721,
        0.2943928627014602, 0.14887433898163122,
        0,
        -0.14887433898163122, -0.2943928627014602,
        -0.43339539412924721, -0.56275713466860466,
        -0.67940956829902444, -0.7808177265864169,
        -0.86506336668898454, -0.93015749135570824,
        -0.97390652851717174, -0.99565716302580809
    ])
}

gauss_weights = {
    15: array([
        0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
        0.381830050505118944950369775488975, 0.417959183673469387755102040816327
    ]),
    21: array([
        0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
        0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
        0.295524224714752870173892994651338
    ])
}

kronrod_weights = {
    15: array([
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714
    ]),
    21: array([
        0.011694638867371874, 0.032558162307964725,
        0.054755896574351995, 0.075039674810919957,
        0.0931254545836976, 0.10938715880229764,
        0.12349197626206584, 0.13470921731147334,
        0.14277593857706009, 0.14773910490133849,
        0.1494455540029169, 0.14773910490133849,
        0.14277593857706009, 0.13470921731147334,
        0.12349197626206584, 0.10938715880229764,
        0.0931254545836976, 0.075039674810919957,
        0.054755896574351995, 0.032558162307964725,
        0.011694638867371874
    ])
}

def get_current_kronrod_weight(nodes_count):
    return append(kronrod_weights[nodes_count], kronrod_weights[nodes_count][:-1][::-1])


def get_current_gausskronrod_nodes(nodes_count):
    return append(gausskronrod_nodes[nodes_count], negative(gausskronrod_nodes[nodes_count][:-1][::-1]))

def get_current_gauss_weight(nodes_count):
    changedGaussWeightArr = append(gauss_weights[nodes_count], gauss_weights[nodes_count][:-1][::-1])

    for item in range(len(changedGaussWeightArr)):
        index = int(item + item + 1)

        changedGaussWeightArr = concatenate((changedGaussWeightArr[:index], [0], changedGaussWeightArr[index:]))

    changedGaussWeightArr = append([0], changedGaussWeightArr)

    return changedGaussWeightArr
