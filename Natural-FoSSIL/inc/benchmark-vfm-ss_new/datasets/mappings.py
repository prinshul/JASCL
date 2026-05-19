import torchvision


def get_cityscapes_mapping():
    return {
        class_.id: class_.train_id
        for class_ in torchvision.datasets.Cityscapes.classes
    }


def get_base_mapping():
    return {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 5,
        6 : 6,
        7 : 7,
        8 : 8,
        9 : 9
    }

def get_step1_mapping():
    return {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 5,
        6 : 6,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14
    }
    
    
def get_step2_mapping():
    return {
        0 : 0,
        1 : 1,
        2 : 2,
        3 : 3,
        4 : 4,
        5 : 5,
        6 : 6,
        7 : 7,
        8 : 8,
        9 : 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19
    }



def get_mapillary_mapping():
    return {
        13: 0,
        24: 0,
        41: 0,
        2: 1,
        15: 1,
        17: 2,
        6: 3,
        3: 4,
        45: 5,
        47: 5,
        48: 6,
        50: 7,
        30: 8,
        29: 9,
        27: 10,
        19: 11,
        20: 12,
        21: 12,
        22: 12,
        55: 13,
        61: 14,
        54: 15,
        58: 16,
        57: 17,
        52: 18,
    }


def get_ade20k_mapping():
    return {i: i - 1 for i in range(1, 151)}
