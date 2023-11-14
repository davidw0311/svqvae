data = {
    'pretrain':{
        'cam16':{
            'paths':
                ['data/CAM16_100cls_10mask/train/data/normal', 
                    'data/CAM16_100cls_10mask/train/data/tumor', 
                    'data/CAM16_100cls_10mask/val/data/normal', 
                    'data/CAM16_100cls_10mask/val/data/tumor',
                    'data/CAM16_100cls_10mask/test/data/normal',
                    'data/CAM16_100cls_10mask/test/data/tumor'],
            'mean': [0.6931, 0.5478, 0.6757],
            'std': [0.1972, 0.2487, 0.1969]
        },
        'prcc' :{
            'paths': 
                ['data/pRCC_nolabel'],
            'mean': [0.6843, 0.5012, 0.6436],
            'std': [0.2148, 0.2623, 0.1969]
        },
    },
    'train':{
        'paths':{
            'wbc_100': 'data/WBC_100/train/data',
            'wbc_50': 'data/train_wbc50/data',
            'wbc_10': 'data/train_wbc10/data',
            'wbc_1': 'data/train_wbc1/data',
            'val': 'data/WBC_100/data'
        },
        'mean': [0.7048, 0.5392, 0.5885],
        'std': [0.1626, 0.1902, 0.0974]
    }   
}