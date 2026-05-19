import torch

seed_current = 1024

BDD_datadir = '<Path_to_Your_Root_Data>/Semi_Supervised_Natural_FoSSIL_data/step1/'
city_datadir = '<Path_to_Your_Root_Data>/Semi_Supervised_Natural_FoSSIL_data/step2/'
IDD_datadir = '<Path_to_Your_Root_Data>/Semi_Supervised_Natural_FoSSIL_data/step3/'
camvid_datadir = '<Path_to_Your_Root_Data>/Semi_Supervised_Natural_FoSSIL_data/step4/'

shots_datadir = '<Path_to_Your_Root_Data>/Semi_Supervised_Natural_FoSSIL_data/shots/'


weights_step1 = torch.tensor([3.6525147,8.799815,4.781908,10.034828,9.556787,
                 8.315293,8.163474,9.246903,6.0067043,9.606205,0])
                              
weights_step2 = torch.tensor([3.6525147,8.799815,4.781908,10.034828,9.556787,
                 8.315293,8.163474,9.246903,6.0067043,9.606205,
                 10.787631,6.842216,0])   

weights_step3 = torch.tensor([3.6525147,8.799815,4.781908,10.034828,9.556787,
                 8.315293,8.163474,9.246903,6.0067043,9.606205,
                 10.787631,6.842216,
                 11.96496,5.440929,0])                     
                 
weights_step4 = torch.tensor([3.6525147,8.799815,4.781908,10.034828,9.556787,
                 8.315293,8.163474,9.246903,6.0067043,9.606205,
                 10.787631,6.842216,
                 11.96496,5.440929, 
                 10.24, 10.324, 10.89, 0])  


def median_frequency_balance(dataset, num_classes, ignore_index=255, _eps=1e-5):
    '''
    For more details refer to Section 6.3.2 in
    https://arxiv.org/pdf/1411.4734.pdf
    '''
    frequency = torch.zeros(num_classes) + _eps
    for _, seg in dataset:
        for cid in torch.unique(seg):
            if cid == ignore_index:
                continue
            frequency[cid] += torch.sum(seg == cid)
    frequency /= torch.sum(frequency)
    return torch.median(frequency) / frequency

